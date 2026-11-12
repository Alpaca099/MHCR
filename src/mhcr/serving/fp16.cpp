// Read-only FP16 shards with saved FP32 inverse norms. No corpus-sized copy.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <immintrin.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
thread_local std::string error;

class Mapping {
    void* data_ = MAP_FAILED;
    size_t bytes_ = 0;
public:
    explicit Mapping(const std::filesystem::path& path) {
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("Cannot open " + path.string());
        struct stat st;
        if (fstat(fd, &st) || st.st_size <= 0) {
            close(fd);
            throw std::runtime_error("Invalid mapping size");
        }
        bytes_ = static_cast<size_t>(st.st_size);
        data_ = mmap(nullptr, bytes_, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        if (data_ == MAP_FAILED) throw std::runtime_error("mmap failed");
        madvise(data_, bytes_, MADV_RANDOM);
    }
    Mapping(const Mapping&) = delete;
    ~Mapping() { if (data_ != MAP_FAILED) munmap(data_, bytes_); }
    const char* data() const { return static_cast<const char*>(data_); }
    size_t bytes() const { return bytes_; }
};

class Store {
    std::unique_ptr<Mapping> norms_;
    std::vector<std::unique_ptr<Mapping>> maps_;
    std::vector<size_t> starts_;
    std::vector<const uint16_t*> shards_;
    const float* inv_;
public:
    size_t n, d;
    explicit Store(const char* manifest) {
        std::ifstream stream(manifest);
        size_t count, offset;
        std::string path;
        if (!(stream >> n >> d >> count >> std::quoted(path) >> offset) ||
            !n || !d || !count || count > n || d % 16 || d > 1024)
            throw std::runtime_error("Invalid FP16 manifest header");
        auto resolve = [&](const std::string& s) {
            std::filesystem::path p(s);
            return p.is_absolute() ? p : std::filesystem::path(manifest).parent_path() / p;
        };
        norms_ = std::make_unique<Mapping>(resolve(path));
        if (offset % 4 || offset > norms_->bytes() ||
            (norms_->bytes() - offset) / 4 != n || (norms_->bytes() - offset) % 4)
            throw std::runtime_error("Inverse norm shape mismatch");
        inv_ = reinterpret_cast<const float*>(norms_->data() + offset);
        size_t first = 0;
        for (size_t i = 0; i < count; ++i) {
            size_t rows;
            if (!(stream >> std::quoted(path) >> offset >> rows) || !rows || rows > n - first)
                throw std::runtime_error("Invalid shard record");
            auto map = std::make_unique<Mapping>(resolve(path));
            if (offset % 2 || offset > map->bytes() ||
                (map->bytes() - offset) / (d * 2) != rows || (map->bytes() - offset) % (d * 2))
                throw std::runtime_error("FP16 shard shape mismatch");
            starts_.push_back(first);
            shards_.push_back(reinterpret_cast<const uint16_t*>(map->data() + offset));
            maps_.push_back(std::move(map));
            first += rows;
        }
        if (first != n || stream >> path) throw std::runtime_error("Incomplete or extra shard records");
    }
    void read(int64_t id, float* output) const {
        if (id < 0 || static_cast<uint64_t>(id) >= n) throw std::out_of_range("Gallery ID out of range");
        size_t s = std::upper_bound(starts_.begin(), starts_.end(), id) - starts_.begin() - 1;
        const uint16_t* src = shards_[s] + (id - starts_[s]) * d;
        const auto scale = _mm512_set1_ps(inv_[id]);
        for (size_t j = 0; j < d; j += 16) {
            auto half = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + j));
            _mm512_storeu_ps(output + j, _mm512_mul_ps(_mm512_cvtph_ps(half), scale));
        }
    }
};

// AVX-512 reduction adapted from RoarGraph's DistanceInnerProduct
// (Copyright (c) 2024 Meng Chen, MIT; see NOTICE). Preserve the experiment's
// 16-lane accumulation and horizontal reduction.
float dot(const float* a, const float* b, size_t d) {
    __m512 sum = _mm512_setzero_ps();
    for (size_t j = 0; j < d; j += 16)
        sum = _mm512_add_ps(sum, _mm512_mul_ps(_mm512_loadu_ps(a + j), _mm512_loadu_ps(b + j)));
    __m256 sum8 = _mm512_extractf32x8_ps(sum, 1);
    sum8 += _mm512_extractf32x8_ps(sum, 0);
    __m128 sum4 = _mm256_extractf128_ps(sum8, 1);
    sum4 += _mm256_extractf128_ps(sum8, 0);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    return _mm_cvtss_f32(sum4);
}
}

extern "C" {
const char* mhcr_half_error() { return error.c_str(); }
void* mhcr_half_open(const char* path) {
    try { return new Store(path); }
    catch (const std::exception& e) { error = e.what(); return nullptr; }
}
void mhcr_half_close(void* handle) { delete static_cast<Store*>(handle); }
uint64_t mhcr_half_size(void* handle) { return static_cast<Store*>(handle)->n; }
uint64_t mhcr_half_dim(void* handle) { return static_cast<Store*>(handle)->d; }
int mhcr_half_read(void* handle, const int64_t* ids, size_t count, float* output) {
    try {
        auto& store = *static_cast<Store*>(handle);
        for (size_t i = 0; i < count; ++i) store.read(ids[i], output + i * store.d);
        return 0;
    } catch (const std::exception& e) { error = e.what(); return -1; }
}
int mhcr_half_scores(void* handle, const float* query, const int64_t* ids, size_t count, float* scores) {
    try {
        auto& store = *static_cast<Store*>(handle);
        alignas(64) float scratch[1024];
        for (size_t i = 0; i < count; ++i) {
            store.read(ids[i], scratch);
            scores[i] = dot(scratch, query, store.d);
        }
        return 0;
    } catch (const std::exception& e) { error = e.what(); return -1; }
}
}
