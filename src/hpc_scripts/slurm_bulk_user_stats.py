#!/usr/bin/env python3
"""Summarize Slurm job CPU and memory usage for a user or a single job."""

from __future__ import annotations

import argparse
import getpass
import os
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional


# ---------- Utilities ----------
def parse_slurm_time_to_seconds(val: str) -> Optional[int]:
    """Parse Slurm time strings like 1-02:03:04 or 02:03:04 to seconds."""
    if not val:
        return None
    val = val.strip()
    if val.lower() in {"unknown", "invalid"}:
        return None
    if val.isdigit():
        return int(val)
    days = 0
    if "-" in val:
        day_part, rest = val.split("-", 1)
        try:
            days = int(day_part)
            val = rest
        except ValueError:
            pass
    parts = val.split(":")
    if len(parts) == 2:
        parts = ["0"] + parts
    if len(parts) < 3:
        return None
    try:
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
    except ValueError:
        return None
    total_seconds = int(days * 86400 + hours * 3600 + minutes * 60 + seconds)
    return total_seconds


def parse_size_to_bytes(s: str) -> Optional[int]:
    """Parse memory strings like '4G', '4000M', '1.5T' to bytes (IEC 1024)."""
    if not s:
        return None
    m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([kKmMgGtTpP]?)", s.strip())
    if not m:
        return None
    val = float(m.group(1))
    unit = (m.group(2) or "").upper()
    table = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4, "P": 1024**5}
    mult = table.get(unit)
    return int(val * mult) if mult else None


def parse_reqmem_to_bytes(mem: str, ncpus: Optional[int], nnodes: Optional[int]) -> Optional[int]:
    """Parse Slurm ReqMem strings (e.g., 4Gn, 4000Mc) into total bytes."""
    if not mem:
        return None
    m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([kKmMgGtTpP]?)([nc]?)\s*$", mem)
    if not m:
        return None
    val = float(m.group(1))
    unit = (m.group(2) or "").upper()
    scope = (m.group(3) or "").lower()  # c=per CPU, n=per node
    base = parse_size_to_bytes(f"{val}{unit}")
    if base is None:
        return None
    total = base
    if scope == "c" and ncpus:
        total = base * ncpus
    elif scope == "n" and nnodes:
        total = base * nnodes
    return int(total)


def fmt_bytes(n: Optional[int]) -> str:
    """Human-readable IEC format."""
    if not n or n <= 0:
        return "n/a"
    for unit, m in [("PiB", 1024**5), ("TiB", 1024**4), ("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)]:
        if n >= m:
            return f"{n/m:.2f} {unit}"
    return f"{n} B"


def secs_to_h(s: Optional[int]) -> str:
    return "n/a" if s is None else f"{s/3600:.2f}"


def pct_str(x: Optional[float]) -> str:
    return f"{x*100:.2f}%" if x is not None else "n/a"


def run(cmd: List[str]) -> str:
    return subprocess.check_output(cmd, text=True, errors="ignore")


def parse_gpus_from_tres(tres: str) -> Optional[int]:
    if not tres:
        return None
    total = 0
    seen = False
    for part in tres.split(","):
        part = part.strip()
        if not part:
            continue
        if part.startswith("gres/gpu=") or part.startswith("gpu="):
            val = part.split("=", 1)[1]
            try:
                total += int(float(val))
                seen = True
            except ValueError:
                pass
    return total if seen else None


def parse_gpus_from_gres(gres: str) -> Optional[int]:
    if not gres:
        return None
    total = 0
    seen = False
    for part in gres.split(","):
        part = part.strip()
        if not part:
            continue
        toks = part.split(":")
        if len(toks) >= 2 and toks[0] == "gpu":
            try:
                count = int(toks[-1])
                total += count
                seen = True
            except ValueError:
                pass
    return total if seen else None


def parse_gpu_util_from_tres_usage(tres_usage: str) -> Optional[float]:
    """
    Parse TRESUsageInAve/Max strings like 'cpu=0,energy=0,gres/gpumem=0,gres/gpuutil=23' and
    return gpuutil percentage as float.
    """
    if not tres_usage:
        return None
    for part in tres_usage.split(","):
        part = part.strip()
        if part.startswith("gres/gpuutil="):
            try:
                return float(part.split("=", 1)[1])
            except ValueError:
                return None
    return None


def parse_gpu_mem_from_tres_usage(tres_usage: str) -> Optional[int]:
    """
    Parse gpumem (MiB) from TRESUsageInAve/Max strings.
    """
    if not tres_usage:
        return None
    for part in tres_usage.split(","):
        part = part.strip()
        if part.startswith("gres/gpumem="):
            try:
                mib = float(part.split("=", 1)[1])
                return int(mib * 1024 * 1024)
            except ValueError:
                return None
    return None


# ---------- sacct parsing ----------
SACCT_FIELDS_BASE = [
    "JobIDRaw",
    "JobName",
    "User",
    "State",
    "ElapsedRaw",
    "TotalCPU",
    "AllocCPUS",
    "ReqMem",
    "MaxRSS",
    "NNodes",
    "NodeList",
]  # minimal set that is widely supported
# Extended field sets (tried in order; sacct may reject unknown fields on some clusters)
SACCT_FIELDS_TRES = SACCT_FIELDS_BASE + ["AllocTRES", "ReqTRES", "TRESUsageInAve", "TRESUsageInMax"]
SACCT_FIELDS_GRES = SACCT_FIELDS_BASE + ["AllocGRES", "ReqGRES", "Gres"]
SACCT_FIELDS_FULL = SACCT_FIELDS_BASE + ["AllocTRES", "ReqTRES", "AllocGRES", "ReqGRES", "Gres", "TRESUsageInAve", "TRESUsageInMax"]
# Default/legacy alias for backward compatibility
SACCT_FIELDS = SACCT_FIELDS_FULL


def parse_state(raw_state: str) -> str:
    """Normalize State values like 'COMPLETED', 'CANCELLED by 0'."""
    if not raw_state:
        return "?"
    return raw_state.split()[0].upper()


def summarize_from_sacct_line(line: str, fields: List[str]) -> Optional[Dict[str, Any]]:
    parts = line.split("|")
    if len(parts) < len(fields):
        return None
    data = {k: parts[i].strip() if i < len(parts) else "" for i, k in enumerate(fields)}
    jobid_raw = data["JobIDRaw"]
    # Skip step records like 123.batch or 123.0
    if not jobid_raw or "." in jobid_raw:
        return None

    jobid = jobid_raw
    name = data["JobName"]
    state = parse_state(data["State"])
    ncpus = int(data["AllocCPUS"]) if data["AllocCPUS"].isdigit() else None
    nnodes = int(data["NNodes"]) if data["NNodes"].isdigit() else None
    ngpus = (
        parse_gpus_from_tres(data.get("AllocTRES", ""))
        or parse_gpus_from_tres(data.get("ReqTRES", ""))
        or parse_gpus_from_gres(data.get("AllocGRES", ""))
        or parse_gpus_from_gres(data.get("ReqGRES", ""))
        or parse_gpus_from_gres(data.get("Gres", ""))
    )
    gpu_util_avg_pct = parse_gpu_util_from_tres_usage(data.get("TRESUsageInAve", ""))
    gpu_util_max_pct = parse_gpu_util_from_tres_usage(data.get("TRESUsageInMax", ""))
    gpu_mem_used_b = parse_gpu_mem_from_tres_usage(data.get("TRESUsageInMax", "")) or parse_gpu_mem_from_tres_usage(data.get("TRESUsageInAve", ""))
    wall_s = int(data["ElapsedRaw"]) if data["ElapsedRaw"].isdigit() else None
    cput_s = parse_slurm_time_to_seconds(data["TotalCPU"])
    used_mem_b = parse_size_to_bytes(data["MaxRSS"])
    req_mem_b = parse_reqmem_to_bytes(data["ReqMem"], ncpus=ncpus, nnodes=nnodes)
    avg_used_cpus = (cput_s / wall_s) if (cput_s is not None and wall_s and wall_s > 0) else None
    cpu_eff = (avg_used_cpus / ncpus) if (avg_used_cpus is not None and ncpus) else None
    mem_eff = (used_mem_b / req_mem_b) if (used_mem_b and req_mem_b and req_mem_b > 0) else None

    return {
        "jobid": jobid,
        "name": name,
        "state": state,
        "nodes": data.get("NodeList") or None,
        "ncpus": ncpus,
        "ngpus": ngpus,
        "gpu_util_avg_pct": gpu_util_avg_pct,
        "gpu_util_max_pct": gpu_util_max_pct,
        "gpu_mem_used_b": gpu_mem_used_b,
        "wall_s": wall_s,
        "cput_s": cput_s,
        "avg_used_cpus": avg_used_cpus,
        "cpu_eff": cpu_eff,
        "used_mem_b": used_mem_b,
        "req_mem_b": req_mem_b,
        "mem_eff": mem_eff,
    }


def list_jobs_with_sacct(user: str, include_finished: bool, jobid: Optional[str]) -> List[Dict[str, Any]]:
    # Short state codes per sacct: PD=pending, R=running, CF=configuring, CG=completing, RQ=requeued, RS=resizing, S=suspended, SO=stageout
    states_active = ["PD", "R", "CF", "CG", "RQ", "RS", "S", "SO"]
    state_variants: List[Optional[str]] = [None] if include_finished else [
        ",".join(states_active),   # full active set
        "PD,R",                    # minimal, broadly supported set
        None,                      # final fallback: no filter (include all)
    ]

    field_variants: List[List[str]] = [
        SACCT_FIELDS_FULL,
        SACCT_FIELDS_TRES,
        SACCT_FIELDS_GRES,
        SACCT_FIELDS_BASE,
    ]

    last_err: Optional[Exception] = None
    out = ""
    fields_used: List[str] = SACCT_FIELDS
    for fields in field_variants:
        for state_arg in state_variants:
            cmd = [
                "sacct",
                "-n",
                "-P",
                f"--format={','.join(fields)}",
            ]
            if user:
                cmd += ["-u", user]
            if state_arg:
                cmd += ["-s", state_arg]
            if jobid:
                cmd += ["-j", jobid]
            try:
                out = run(cmd)
                last_err = None
                fields_used = fields
                break
            except subprocess.CalledProcessError as e:
                last_err = e
                continue
        if last_err is None:
            break

    if last_err:
        raise last_err

    rows = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        r = summarize_from_sacct_line(line, fields_used)
        if r:
            rows.append(r)
    return rows


# ---------- Output ----------
def render_table(rows: List[Dict[str, Any]], name_max: int) -> None:
    cols = ["JOBID", "STATE", "NAME", "NODES", "NCPUS", "NGPUS", "WALL(h)", "CPUT(h)", "avgCPU", "CPUeff", "memUsed", "memReq", "memEff"]
    w = {c: len(c) for c in cols}
    table = []
    for r in rows:
        name = r.get("name") or ""
        if name_max > 0 and len(name) > name_max:
            name = name[: max(0, name_max - 1)] + "…"
        row = {
            "JOBID": r["jobid"],
            "STATE": r["state"],
            "NAME": name,
            "NODES": r.get("nodes") or "n/a",
            "NCPUS": str(r["ncpus"] if r["ncpus"] is not None else "n/a"),
            "NGPUS": str(r["ngpus"] if r.get("ngpus") is not None else "n/a"),
            "WALL(h)": secs_to_h(r["wall_s"]),
            "CPUT(h)": secs_to_h(r["cput_s"]),
            "avgCPU": f"{r['avg_used_cpus']:.2f}" if r["avg_used_cpus"] is not None else "n/a",
            "CPUeff": pct_str(r["cpu_eff"]),
            "memUsed": fmt_bytes(r["used_mem_b"]),
            "memReq": fmt_bytes(r["req_mem_b"]),
            "memEff": pct_str(r["mem_eff"]),
        }
        for k, v in row.items():
            w[k] = max(w[k], len(str(v)))
        table.append(row)
    hdr = "  ".join(f"{c:<{w[c]}}" for c in cols)
    print(hdr)
    print("-" * len(hdr))
    for row in table:
        print("  ".join(f"{row[c]:<{w[c]}}" for c in cols))


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    import csv

    fields = [
        "jobid",
        "name",
        "state",
        "nodes",
        "ncpus",
        "ngpus",
        "gpu_util_avg_pct",
        "gpu_util_max_pct",
        "gpu_mem_used_b",
        "gpu_mem_used_gb",
        "wall_s",
        "cput_s",
        "avg_used_cpus",
        "cpu_eff",
        "used_mem_b",
        "used_mem_gb",
        "req_mem_b",
        "req_mem_gb",
        "mem_eff",
    ]
    f = sys.stdout if path == "-" else open(path, "w", newline="")
    with f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for r in rows:
            row = {
                "jobid": r.get("jobid"),
                "name": r.get("name"),
                "state": r.get("state"),
                "nodes": r.get("nodes"),
                "ncpus": r.get("ncpus"),
                "ngpus": r.get("ngpus"),
                "gpu_util_avg_pct": r.get("gpu_util_avg_pct"),
                "gpu_util_max_pct": r.get("gpu_util_max_pct"),
                "gpu_mem_used_b": r.get("gpu_mem_used_b"),
                "wall_s": r.get("wall_s"),
                "cput_s": r.get("cput_s"),
                "avg_used_cpus": r.get("avg_used_cpus"),
                "cpu_eff": r.get("cpu_eff"),
                "used_mem_b": r.get("used_mem_b"),
                "req_mem_b": r.get("req_mem_b"),
                "mem_eff": r.get("mem_eff"),
            }

            if row["avg_used_cpus"] is not None:
                row["avg_used_cpus"] = round(row["avg_used_cpus"], 2)

            for field in ["cpu_eff", "mem_eff"]:
                if row.get(field) is not None:
                    row[field] = round(row[field], 2)

            for src, dest in [
                ("used_mem_b", "used_mem_gb"),
                ("req_mem_b", "req_mem_gb"),
                ("gpu_mem_used_b", "gpu_mem_used_gb"),
            ]:
                val = row.get(src)
                row[dest] = round((val / (1024**3)), 2) if val is not None else None

            w.writerow({k: row.get(k) for k in fields})


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    import math

    def mean(xs):
        xs = [x for x in xs if x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))]
        return sum(xs) / len(xs) if xs else float("nan")

    peak_mem = None
    mem_vals = [r.get("used_mem_b") for r in rows if r.get("used_mem_b")]
    if mem_vals:
        peak_mem = max(mem_vals)
    unique_nodes: set[str] = set()
    state_counts: Dict[str, int] = {}
    for r in rows:
        nodes_field = r.get("nodes")
        if not nodes_field:
            pass
        else:
            for node in nodes_field.split(","):
                node = node.strip()
                if node:
                    unique_nodes.add(node)

        state = (r.get("state") or "").strip().upper() or "?"
        state_counts[state] = state_counts.get(state, 0) + 1

    other_states = {
        k: v
        for k, v in state_counts.items()
        if k not in {"RUNNING", "R", "PENDING", "PD", "COMPLETED", "CD"}
    }
    gpu_util_avgs = [r.get("gpu_util_avg_pct") for r in rows if r.get("gpu_util_avg_pct") is not None]
    gpu_hours_used = []
    gpu_hours_req = []
    for r in rows:
        ng = r.get("ngpus")
        wall = r.get("wall_s")
        util = r.get("gpu_util_avg_pct")
        if ng and wall:
            gpu_hours_req.append(ng * wall / 3600.0)
            if util is not None:
                gpu_hours_used.append((util / 100.0) * ng * wall / 3600.0)

    return {
        "jobs": len(rows),
        "avg_CPUeff_%": mean([r["cpu_eff"] * 100 for r in rows if r.get("cpu_eff") is not None]),
        "avg_avgCPU": mean([r["avg_used_cpus"] for r in rows if r.get("avg_used_cpus") is not None]),
        "avg_memEff_%": mean([r["mem_eff"] * 100 for r in rows if r.get("mem_eff") is not None]),
        "max_mem_b": peak_mem,
        "unique_nodes": len(unique_nodes),
        "state_counts": state_counts,
        "state_other_total": sum(other_states.values()),
        "state_other_breakdown": other_states,
        "avg_gpu_util_pct": mean(gpu_util_avgs),
        "gpu_hours_used": sum(gpu_hours_used) if gpu_hours_used else float("nan"),
        "gpu_hours_req": sum(gpu_hours_req) if gpu_hours_req else float("nan"),
    }


# ---------- CLI ----------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Slurm job CPU and memory statistics (single job or all jobs for a user)."
    )
    ap.add_argument(
        "--job",
        help="Job ID to summarize (default: $SLURM_JOB_ID). Returns the current job statistics if valid $SLURM_JOB_ID is found and --user is not set.",
    )
    ap.add_argument(
        "--user",
        help="Summarize all jobs of USER (default: current user). The --user argument takes precedence over --job.",
    )
    ap.add_argument(
        "--include-finished",
        action="store_true",
        help="Include finished jobs (sacct --state=all). Without this flag only pending/running jobs are shown.",
    )
    ap.add_argument(
        "--csv",
        metavar="PATH",
        help='Write CSV to PATH (use "-" for stdout)',
    )
    ap.add_argument(
        "--name-max",
        type=int,
        default=30,
        help="Max width for job name column; 0=disable truncation (default: 30)",
    )
    args = ap.parse_args()

    if not shutil.which("sacct"):
        print("ERROR: sacct not found in PATH.", file=sys.stderr)
        sys.exit(1)

    rows: List[Dict[str, Any]] = []

    jobid = args.job or os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
    default_user = (
        os.environ.get("SLURM_JOB_USER") or os.environ.get("USER") or getpass.getuser()
    )

    if args.user is None and jobid:
        rows = list_jobs_with_sacct(user="", include_finished=True, jobid=jobid)
    else:
        user = args.user or default_user
        if not user:
            print(
                "Provide --job JOBID or run inside Slurm with $SLURM_JOB_ID set, or use --user USER.",
                file=sys.stderr,
            )
            sys.exit(2)
        rows = list_jobs_with_sacct(user=user, include_finished=args.include_finished, jobid=None)

    render_table(rows, name_max=args.name_max)
    agg = aggregate(rows)
    print("\nSummary:")
    print(f"  jobs:         {agg['jobs']}")
    print(f"  unique nodes: {agg['unique_nodes']}")
    state_counts = agg.get("state_counts", {})
    r_count = sum(state_counts.get(k, 0) for k in ("RUNNING", "R"))
    pd_count = sum(state_counts.get(k, 0) for k in ("PENDING", "PD"))
    cd_count = sum(state_counts.get(k, 0) for k in ("COMPLETED", "CD"))
    other_total = agg.get("state_other_total", 0)
    print(f"  states:       R={r_count}  PD={pd_count}  CD={cd_count}  other={other_total}")
    other_breakdown = agg.get("state_other_breakdown") or {}
    if other_breakdown:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(other_breakdown.items()))
        print(f"    other breakdown: {detail}")
    if agg["avg_CPUeff_%"] == agg["avg_CPUeff_%"]:
        print(f"  mean CPUeff:  {agg['avg_CPUeff_%']:.2f}%")
    if agg["avg_avgCPU"] == agg["avg_avgCPU"]:
        print(f"  mean avgCPU:  {agg['avg_avgCPU']:.2f}")
    if agg["avg_memEff_%"] == agg["avg_memEff_%"]:
        print(f"  mean memEff:  {agg['avg_memEff_%']:.2f}%")
    if agg.get("max_mem_b"):
        print(f"  max memUsed:  {fmt_bytes(agg['max_mem_b'])}")
    if agg.get("avg_gpu_util_pct") == agg.get("avg_gpu_util_pct"):
        print(f"  mean GPU util: {agg['avg_gpu_util_pct']:.2f}%")
    if agg.get("gpu_hours_req") == agg.get("gpu_hours_req"):
        used = agg.get("gpu_hours_used")
        if used == used:  # not NaN
            print(f"  GPU hours:    used {used:.2f} / requested {agg['gpu_hours_req']:.2f}")
        else:
            print(f"  GPU hours:    requested {agg['gpu_hours_req']:.2f} (util data unavailable)")

    if args.csv:
        write_csv(rows, args.csv)


if __name__ == "__main__":
    main()
