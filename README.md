# hpc-scripts

Utilities for working with high-performance computing (HPC) environments. The scripts
help inspect PBS/Slurm job efficiency and monitor CPU and memory usage on a
running system or process tree.

Made with Codex help :)

## Dependencies

Install the required Python packages with pip:

| Feature | Packages | Install command |
| ------- | -------- | ---------------- |
| Core utilities | psutil | `pip install psutil` |
| Plotting for `psutil-monitor` | matplotlib, numpy | `pip install matplotlib numpy` |

The `pbs-bulk-user-stats` command also expects the PBS `qstat` utility to be
available in your environment.
The `slurm-bulk-user-stats` command expects Slurm's `sacct` utility to be
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

Examples:

```bash
# Summarize a specific job and write CSV output
pbs-bulk-user-stats --job 12345 --csv stats.csv

# Summarize all jobs for the current user (default) 
pbs-bulk-user-stats --include-finished

# Summarize all jobs for a specific user
pbs-bulk-user-stats --user myuser --include-finished
```

When invoked with no `--user` or `--job` options:
- On a login node (no `$PBS_JOBID` present), it summarizes all jobs for the current user.
- Inside a running PBS job (where `$PBS_JOBID` is set), it automatically summarizes that specific job.

```
pbs-bulk-user-stats
```

State codes (PBS):
- `R` running, `Q` queued/waiting, `X` finished (requires `--include-finished`), other codes are printed under “other” in the summary.

**Expected output:**
```
$ pbs-bulk-user-stats

JOBID    STATE   NAME       NODES    NCPUS  WALL(h)  CPUT(h)  avgCPU  CPUeff  memUsed   memReq   memEff
-------------------------------------------------------------------------------------------------------
0001      R      run1		pbs-1    176    38.55    3632.12  163.6  93.53%  207.4 GiB 256.00 GiB 81.10%
0002      R      run2		pbs-2    176    38.59    3589.72  93.13  52.91%  50.02 GiB 256.00 GiB 19.54%
...
Summary:
  jobs:         5
  unique nodes: 3
  states:       R=4  Q=1  X=0  other=0
  mean CPUeff:  75.20%
  mean avgCPU:  132.35
  mean memEff:  82.50%
  max memUsed:  230.16 GiB

```
or if run inside a running PBS:
```
JOBID  STATE  NAME   NODES  NCPUS  WALL(h)  CPUT(h)  avgCPU  CPUeff  memUsed     memReq     memEff
-----------------------------------------------------------------------------------------------------
0001   R      STDIN  pbs-5  100    0.03     0.01     0.22    0.22%   666.58 MiB  30.00 GiB  2.17% 

Summary:
  jobs:        1
  mean CPUeff: 0.22%
  mean avgCPU: 0.22
  mean memEff: 2.17%
  max memUsed: 666.58 MiB

```

After the table, a summary reports the job count, mean CPU efficiency,
mean average CPU usage, mean memory efficiency, and the peak memory used
across all listed jobs.

### `psutil-monitor`

Real-time CPU and memory monitor for the system or a process tree.

Examples:

```bash
# System-wide (by default) monitoring with console output only
psutil-monitor

# System-wide monitoring with CSV and PNG output
psutil-monitor --mode system --csv node.csv --plot node.png

# Monitor the current process tree (useful inside a PBS job)
psutil-monitor --mode proc --pid $$ --include-children --csv job.csv

# For script.py resources monitoring:
python script.py &                   # launch the workload
target=$!                            # PID of script.py
echo $target
# psutil-monitor watches that PID and exits when the process tree is gone
psutil-monitor --mode proc --pid "$target" --include-children --csv stat.csv --plot plot.png

```
**Expected output:**
```
$ psutil-monitor

CPUs available (affinity): 384
Total memory available: 754.76 GiB
CPU basis for %: 384
Memory basis for %: 754.76 GiB
2025-08-14T15:20:14  CPU  79.67%  busyCPUs 305.93  (provided 384)  MEM   9.93%  used 74.96 GiB / total 754.76 GiB
2025-08-14T15:20:16  CPU  69.30%  busyCPUs 266.13  (provided 384)  MEM   9.95%  used 75.12 GiB / total 754.76 GiB
2025-08-14T15:20:18  CPU  61.34%  busyCPUs 235.53  (provided 384)  MEM  10.05%  used 75.82 GiB / total 754.76 GiB
2025-08-14T15:20:20  CPU  61.32%  busyCPUs 235.47  (provided 384)  MEM  10.09%  used 76.15 GiB / total 754.76 GiB
2025-08-14T15:20:22  CPU  74.57%  busyCPUs 286.33  (provided 384)  MEM   9.94%  used 74.99 GiB / total 754.76 GiB
2025-08-14T15:20:24  CPU  85.94%  busyCPUs 330.01  (provided 384)  MEM   9.86%  used 74.44 GiB / total 754.76 GiB
Average busy CPUs over run: 276.570
Peak memory (system): 76.15 GiB

```

Use the `--help` option of each command to see all available options.

### `slurm-bulk-user-stats`

Summarize CPU and memory usage for Slurm jobs and show which nodes the jobs are
allocated to. The command relies on `sacct` being available in your `PATH`.

State codes (Slurm):
- `R`/`RUNNING`, `PD`/`PENDING`, `CD`/`COMPLETED`; other states (e.g., `F`, `CG`, `S`, `TO`) are grouped under “other” in the summary and listed in the breakdown.

Examples:

```bash
# Summarize a specific job and write CSV output
slurm-bulk-user-stats --job 12345 --csv stats.csv

# Summarize all running jobs for the current user (default)
slurm-bulk-user-stats

# Summarize all jobs (including finished) for a specific user
slurm-bulk-user-stats --user myuser --include-finished
```

When invoked with no `--user` or `--job` options:
- On a login node (no `$SLURM_JOB_ID` present), it summarizes pending/running jobs for the current user.
- Inside a running Slurm job (where `$SLURM_JOB_ID` is set), it automatically summarizes that specific job.

```
slurm-bulk-user-stats
```

The output mirrors the PBS version, showing job state, node list, CPU/memory
usage, efficiency metrics, and a summary block with job counts and averages.
