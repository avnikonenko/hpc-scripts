# Changelog

## 0.1.0
- Initial release of `hpc-scripts` with PBS/Slurm job summaries and psutil-based monitoring.

## 0.2 (Unreleased)
- Added GPU request counts (NGPUS) to PBS and Slurm bulk user stats outputs and CSVs.
- Added README note for installing from GitHub with plotting extras.
- Added optional GPU monitoring to psutil-monitor via `--gpu` (NVML/pynvml), including terminal and CSV outputs.
- Added pip extras `gpu` and `all` (plot + GPU) for simpler installs.

## 0.1.1
- PBS and Slurm support
- Added job state counts (running/pending/finished/other) to PBS and Slurm bulk user stats summaries.
