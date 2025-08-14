# hpc-scripts

Utilities for working with high-performance computing (HPC) environments. The scripts
help inspect PBS job efficiency and monitor CPU and memory usage on a running
system or process tree.

## Dependencies

Install the required Python packages with pip:

| Feature | Packages | Install command |
| ------- | -------- | ---------------- |
| Core utilities | psutil | `pip install psutil` |
| Plotting for `psutil-monitor` | matplotlib, numpy | `pip install matplotlib numpy` |

The `pbs-bulk-user-stats` command also expects the PBS `qstat` utility to be
available in your environment.

## Installation

Clone the repository and install with pip:

```bash
pip install .            # core utilities
# or include plotting support for psutil-monitor
pip install .[plot]
```

The base installation depends on [psutil](https://pypi.org/project/psutil/).
The optional `plot` extra pulls in `matplotlib` and `numpy` for the `--plot`
feature of `psutil-monitor`.

## CLI tools

### `pbs-bulk-user-stats`

Summarize CPU and memory usage for PBS jobs and show which nodes the jobs are
allocated to. The command relies on `qstat` being available in your `PATH`.
CSV output rounds the `avg_used_cpus` column and includes memory fields in both
bytes and gigabytes.

Examples:

```bash
# Summarize a specific job and write CSV output
pbs-bulk-user-stats --job 12345 --csv stats.csv 

# Summarize all jobs for a user
pbs-bulk-user-stats --user myuser --include-finished
```

### `psutil-monitor`

Real-time CPU and memory monitor for the system or a process tree.

Examples:

```bash
# System-wide (by default) monitoring with CSV and PNG output
psutil-monitor --mode system --csv node.csv --plot node.png
psutil-monitor 

# Monitor the current process tree (useful inside a PBS job)
psutil-monitor --mode proc --pid $$ --include-children --csv job.csv
```

Use the `--help` option of each command to see all available options.
