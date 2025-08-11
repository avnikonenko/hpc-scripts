# hpc-scripts

Utilities for working with high-performance computing (HPC) environments. The scripts
help inspect PBS job efficiency and monitor CPU and memory usage on a running
system or process tree.

## Installation

Clone the repository and install with pip:

```bash
pip install .
```

This will provide the following command line tools:

- `pbs-bulk-user-stats` – summarize CPU and memory usage for PBS jobs.
- `psutil-monitor` – real-time CPU and memory monitor for the system or a process tree.

See the `--help` option of each command for details and examples.
