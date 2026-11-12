# Memory measurement

`mhcr.storage.memory()` reports the current process's lifetime peak RSS in bytes.
Index build status records this value after each populated snapshot. This is
not an aggregate over multiple resumed processes and may precede final FastScan
conversion. Use a fresh process and an external monitor such as `/usr/bin/time -v`
to measure the complete build, including serialization and final conversion.

Measure online RSS in a separate process after index/model loading and search.
Training peaks include training features and labels; report them separately.
Retained snapshots consume disk space, not necessarily resident memory. Budget
disk space for all snapshots and memory for conversion/loading copies before
running a full dataset. The portable builder is not certified to fit 128M on a
particular machine merely because its configuration covers the full gallery.
