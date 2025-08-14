#!/usr/bin/env python3
"""
psutil_monitor.py — real-time CPU & Memory monitor

Features
- Mode "system": overall node CPU% and RAM usage
- Mode "proc": aggregate CPU% and RSS of a PID + all children (multiprocessing-friendly)
- Prints live metrics, writes CSV, optional PNG plot at the end
- Reports **busy CPU equivalents** per sample (e.g., 6.4 means ~6.4 cores busy)
- Prints **overall average busy CPUs** at the end
- Uses CPU affinity (os.sched_getaffinity) as the default CPU basis (overridable by PBS_NP or --ncpu-basis)
- Uses total memory as the default memory basis (overridable by --mem-basis in GB)
- Prints the detected CPU set, total memory, and chosen bases at startup

Examples
  # System-wide monitoring, 2s interval, save CSV + PNG
  python3 psutil_monitor.py --mode system --interval 2 --csv node.csv --plot node.png

  # Monitor this Python process tree (useful inside a PBS job)
  python3 psutil_monitor.py --mode proc --pid $$ --include-children --csv job.csv --plot job.png

  # Monitor an existing PID (e.g., your job's launcher), 1s interval, no plot
  python3 psutil_monitor.py --mode proc --pid 12345 --include-children --interval 1 --csv job.csv
"""
import argparse
import csv
import datetime as dt
import os
import signal
import sys
import time
from typing import Dict, Optional

import psutil

# ---------- Helpers ----------
def bytes_human(n: int) -> str:
    if n is None:
        return "n/a"
    for unit, m in [("PiB",1024**5),("TiB",1024**4),("GiB",1024**3),("MiB",1024**2),("KiB",1024)]:
        if n >= m:
            return f"{n/m:.2f} {unit}"
    return f"{n} B"

def now_iso():
    return dt.datetime.now().isoformat(timespec="seconds")

# ---------- Process-tree aggregation (for --mode proc) ----------
def proc_tree_cpu_mem(pid: int, prev_cpu: Dict[int,float]) -> tuple[float, int, int, Dict[int,float], int]:
    """
    Returns per-sample deltas/aggregates for a process tree:
      (cpu_seconds_delta, rss_bytes_sum, proc_count, new_prev_cpu, alive_root_pid)
    """
    try:
        root = psutil.Process(pid)
        alive_root = root.is_running() and root.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return (0.0, 0, 0, {}, 0)

    procs = [root]
    try:
        procs.extend(root.children(recursive=True))
    except psutil.Error:
        pass

    new_prev: Dict[int,float] = {}
    cpu_delta = 0.0
    rss_sum = 0
    count = 0

    for p in procs:
        try:
            if not p.is_running():
                continue
            ct = p.cpu_times()
            used = float(getattr(ct, "user", 0.0) + getattr(ct, "system", 0.0))
            prev = prev_cpu.get(p.pid, used)
            if used >= prev:
                cpu_delta += (used - prev)
            new_prev[p.pid] = used
            with p.oneshot():
                mem = p.memory_info().rss
            rss_sum += int(mem)
            count += 1
        except psutil.Error:
            continue

    return (cpu_delta, rss_sum, count, new_prev, (root.pid if alive_root else 0))

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Real-time CPU & memory monitor (system or process-tree) with CSV + optional plot.")
    ap.add_argument("--mode", choices=["system","proc"], default="system", help="What to monitor")
    ap.add_argument("--pid", type=int, help="Root PID for --mode proc (defaults to current process)")
    ap.add_argument("--include-children", action="store_true", help="Sum over children recursively (recommended for multiprocessing)")
    ap.add_argument("--interval", type=float, default=2.0, help="Sampling interval in seconds (default: 2)")
    ap.add_argument("--duration", type=float, default=0.0, help="Stop after N seconds (0 = run until Ctrl+C)")
    ap.add_argument("--csv", default="monitor.csv", help="CSV output path (default: monitor.csv)")
    ap.add_argument("--plot", default="", help="PNG output path (optional; requires matplotlib)")
    ap.add_argument("--ncpu-basis", type=int, default=0, help="CPU basis for %% calculations (defaults: PBS_NP or logical CPU count)")
    ap.add_argument("--mem-basis", type=float, default=0.0, help="Memory basis for %% calculations (defaults: total memory) in GB")
    args = ap.parse_args()

    # Resolve CPU affinity and bases
    try:
        affinity_cpus = sorted(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-Linux platforms
        affinity_cpus = list(range(psutil.cpu_count(logical=True) or 1))
    ncpu_affinity = len(affinity_cpus)
    ncpu_env = int(os.getenv("PBS_NP", "0"))
    ncpu_basis = args.ncpu_basis or ncpu_env or ncpu_affinity

    # Print detected resources
    print(f"CPUs available (affinity): {ncpu_affinity}")
    vm_total = psutil.virtual_memory().total
    mem_basis = int(args.mem_basis * (1024**3)) if args.mem_basis else vm_total
    print(f"Total memory available: {bytes_human(vm_total)}")
    print(f"CPU basis for %: {ncpu_basis}")
    print(f"Memory basis for %: {bytes_human(mem_basis)}")

    # Prepare CSV (add busy_cpus column)
    fields = [
        "ts",
        "mode",
        "cpu_percent",
        "busy_cpus",
        "mem_percent",
        "mem_used_bytes",
        "mem_used_gb",
        "proc_count",
        "provided_cpus",
        "provided_mem_bytes",
        "provided_mem_gb",
    ]
    f = open(args.csv, "w", newline="")
    w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    w.writeheader(); f.flush()

    # State for graceful shutdown + proc tracking
    stop = False
    def _sigint(_sig, _frm):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    prev_cpu: Dict[int,float] = {}
    rss_peak = 0
    t0 = time.time()

    # Running average of busy CPUs
    samples = 0
    busy_sum = 0.0

    # Warm-up for system CPU so first value isn’t a dummy
    if args.mode == "system":
        psutil.cpu_percent(interval=None)

    # Resolve PID for proc mode
    target_pid: Optional[int] = None
    if args.mode == "proc":
        target_pid = args.pid or os.getpid()
        # Initialize prev_cpu snapshot for more accurate first delta
        try:
            root = psutil.Process(target_pid)
            plist = [root]
            if args.include_children:
                plist += root.children(recursive=True)
            for p in plist:
                try:
                    ct = p.cpu_times()
                    prev_cpu[p.pid] = float(ct.user + ct.system)
                except psutil.Error:
                    pass
        except psutil.Error:
            print("ERROR: target PID not found or inaccessible.", file=sys.stderr)
            sys.exit(1)

    # Buffers for optional plotting
    times, cpu_series, mem_series = [], [], []

    # --- Main loop ---
    while not stop:
        loop_start = time.time()

        if args.mode == "system":
            # cpu_percent blocks for 'interval' to compute average over that window
            percpu = psutil.cpu_percent(interval=args.interval, percpu=True)
            allowed = [percpu[i] for i in affinity_cpus if i < len(percpu)]
            busy_cpus = sum(allowed) / 100.0  # average busy cores over the interval
            cpu_pct = (100.0 * busy_cpus / ncpu_basis) if ncpu_basis else 0.0
            vm = psutil.virtual_memory()
            mem_used = int(vm.used)
            mem_pct = (100.0 * mem_used / mem_basis) if mem_basis else 0.0
            pcount = 0
        else:
            # proc mode: measure delta CPU seconds over interval
            time.sleep(args.interval)
            delta_cpu, rss_sum, pcount, prev_cpu, alive_pid = proc_tree_cpu_mem(target_pid, prev_cpu)
            if alive_pid == 0 and pcount == 0:
                # process tree is gone
                stop = True
            # Convert delta CPU seconds to busy core-equivalents and percent
            busy_cpus = (delta_cpu / args.interval) if args.interval > 0 else 0.0
            cpu_pct = (100.0 * busy_cpus / ncpu_basis) if ncpu_basis > 0 else 0.0
            # Memory: process RSS sum; percent vs basis
            mem_used = int(rss_sum)
            mem_pct = (100.0 * rss_sum / mem_basis) if mem_basis else 0.0
            rss_peak = max(rss_peak, rss_sum)

        # Update running average
        samples += 1
        busy_sum += busy_cpus

        # Print line in canonical format
        provided_cpus = ncpu_basis
        provided_mem = mem_basis
        print(
            f"{now_iso()}  CPU {cpu_pct:6.2f}%  busyCPUs {busy_cpus:6.2f}  (provided {provided_cpus})  "
            f"MEM {mem_pct:6.2f}%  used {bytes_human(mem_used)} / total {bytes_human(provided_mem)}"
            + (f"  procs={pcount}" if args.mode == "proc" else "")
        )
        sys.stdout.flush()

        # Log to CSV
        w.writerow({
            "ts": now_iso(),
            "mode": args.mode,
            "cpu_percent": f"{cpu_pct:.2f}",
            "busy_cpus": f"{busy_cpus:.3f}",
            "mem_percent": f"{mem_pct:.2f}",
            "mem_used_bytes": mem_used,
            "mem_used_gb": f"{mem_used / (1024**3):.3f}",
            "proc_count": pcount if args.mode == "proc" else "",
            "provided_cpus": provided_cpus,
            "provided_mem_bytes": provided_mem,
            "provided_mem_gb": f"{provided_mem / (1024**3):.3f}",
        })
        f.flush()

        # Save for plotting
        times.append(time.time())
        cpu_series.append(cpu_pct)
        mem_series.append(mem_pct)

        # Duration check
        if args.duration > 0 and (time.time() - t0) >= args.duration:
            stop = True

        # Keep loop cadence in system mode (cpu_percent already slept)
        if args.mode == "system":
            # Optional extra sleep if cpu_percent took less than interval (it usually doesn't)
            elapsed = time.time() - loop_start
            if elapsed < args.interval:
                time.sleep(max(0.0, args.interval - elapsed))

    f.close()

    # Optional plot
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            if len(times) >= 2:
                t0 = times[0]
                tmins = [(t - t0)/60.0 for t in times]  # minutes
                fig = plt.figure(figsize=(9,3.5))
                ax = plt.gca()
                ax.plot(tmins, cpu_series, linewidth=1.2, label="CPU %")
                ax.set_xlabel("Time (min)")
                ax.set_ylabel("CPU %")
                ax.set_ylim(0, max(100.0, max(cpu_series) if cpu_series else 100.0))
                ax.grid(True, linestyle="--", alpha=0.4)
                # Add memory on twin axis
                ax2 = ax.twinx()
                ax2.plot(tmins, mem_series, linewidth=1.0, linestyle=":", label="Mem %")
                ax2.set_ylabel("Memory %")
                ax2.set_ylim(0, max(100.0, max(mem_series) if mem_series else 100.0))
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
                plt.title(
                    f"{args.mode} monitor ({ncpu_basis} CPUs, {bytes_human(mem_basis)} memory basis)"
                )
                plt.tight_layout()
                plt.savefig(args.plot, dpi=150)
                print(f"Saved plot: {args.plot}")
            else:
                print("Not enough samples to plot.")
        except Exception as e:
            print(f"Plotting failed: {e}. CSV is still saved at {args.csv}", file=sys.stderr)

    # Final notes
    if samples > 0:
        avg_busy = busy_sum / samples
        print(f"Average busy CPUs over run: {avg_busy:.3f}")
    if args.mode == "proc" and rss_peak:
        print(f"Peak RSS (proc tree): {bytes_human(rss_peak)}")

if __name__ == "__main__":
    main()
