# Training and evaluation protocol

The teacher uses rank-exponential GT100 weights with gamma20. The student uses
uniform GT100 cell coverage: every true neighbor contributes one vote. Its loss
is `10 * budget_primary + 0.1 * uniform_CE + 0.025 * teacher_CE`.
Budget-tail averages mean missing mass with the mean of the worst20% of queries
in a batch; it is not a differentiable P99 latency loss.

The student starts from a PCA projection of the teacher's normalized prototypes
and encoder. Hidden-width reduction retains units selected by weight importance.
The projection basis is persisted so resumed training does not recompute it.
Teacher and student save optimizer, scheduler and RNG state at each epoch.

Scan costs are measured using training queries only, with separate training-query
IDs held out for cost-fit diagnostics. A nonnegative fit separates fixed overhead,
per-list work and padded PQ-block work. Only marginal costs enter routing.
Training budget ranges are derived from the projected router's coverage of
training neighbors and saved in the resolved configuration.

DataComp128M uses a millisecond budget and a12-step implicit gate. The other
recipes use fractions of gallery entries and a32-step implicit gate. Costs and
budgets must remain in matching units; ranges from another index or machine are
not interchangeable.

The query engine scores coverage, orders cells by coverage-to-cost density and
chooses a prefix. It visits at least one cell and extends the selection if fewer
than100 gallery entries are available. FastScan returns candidates for refinement
against the canonical vectors. DataComp decodes FP16 storage to canonical FP32
using saved inverse norms, without a full FP32 corpus copy.

The pipeline searches the configured penalty, nprobe and refinement grids using
returned-ID Recall@100 on calibration queries. For each target and method it
selects the fastest measured feasible point, freezes it, and evaluates disjoint
test queries. An unattained target is recorded as unavailable; calibration
recall is never substituted for test recall. The individual `mhcr-evaluate`
penalty-calibration option instead selects the largest feasible tested penalty
for its one fixed refinement budget.

New runs create their own partitions, PQ codes, models and cost arrays. Seeds,
input identities, package-source hashes, numerical-library versions and stage
receipts are saved alongside outputs. Floating-point libraries and CPU/GPU
partition training can produce different models; bitwise inference comparisons
require identical frozen assets and numerical conventions.
