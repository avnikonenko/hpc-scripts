#!/usr/bin/env python3
"""Summarize PBS job CPU and memory usage for a user or single job."""

import argparse
import os
import re
import shutil
import subprocess
import sys
from typing import Optional, Dict, Any, List

# ---------- Regex patterns for qstat -fx output ----------
RE_CPUT    = re.compile(r"^\s*resources_used\.cput\s*=\s*(\S+)", re.M)
RE_WALL    = re.compile(r"^\s*resources_used\.walltime\s*=\s*(\S+)", re.M)
RE_MEMU    = re.compile(r"^\s*resources_used\.mem\s*=\s*(\S+)", re.M)
RE_VMEMU   = re.compile(r"^\s*resources_used\.vmem\s*=\s*(\S+)", re.M)
RE_NCPUS   = re.compile(r"^\s*Resource_List\.ncpus\s*=\s*(\d+)", re.M)
RE_MEMREQ  = re.compile(r"^\s*Resource_List\.mem\s*=\s*(\S+)", re.M)
RE_VMEMRQ  = re.compile(r"^\s*Resource_List\.(?:vmem|pvmem)\s*=\s*(\S+)", re.M)
RE_SELECT  = re.compile(r"^\s*Resource_List\.select\s*=\s*([^\s]+)", re.M)
RE_OWNER   = re.compile(r"^\s*Job_Owner\s*=\s*([^@]+)@", re.M)
RE_STATE   = re.compile(r"^\s*job_state\s*=\s*(\S+)", re.M)
RE_NAME    = re.compile(r"^\s*Job_Name\s*=\s*(\S+)", re.M)
RE_JOBID   = re.compile(r"^\s*Job Id:\s*(\S+)", re.M)
RE_EXECHOST = re.compile(r"^\s*exec_host\s*=\s*(\S.*(?:\n\s+\S.*)*)", re.M)

# ---------- Utilities ----------
def hms_to_seconds(hms: str) -> int:
    """Convert HHHH:MM:SS (hours may exceed 24) to seconds."""
    parts = [int(x) for x in hms.strip().split(":")]
    sec, mult = 0, 1
    for x in reversed(parts):
        sec += x * mult
        mult *= 60
    return sec

def parse_size_to_bytes(s: str) -> Optional[int]:
    """Parse '123456kb', '64gb', '1.5GB', '2048' to bytes (IEC 1024)."""
    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z]*)$", s.strip())
    if not m:
        return None
    val = float(m.group(1))
    unit = (m.group(2) or "").lower()
    table = {
        "":1, "b":1,
        "k":1024, "kb":1024, "kib":1024,
        "m":1024**2, "mb":1024**2, "mib":1024**2,
        "g":1024**3, "gb":1024**3, "gib":1024**3,
        "t":1024**4, "tb":1024**4, "tib":1024**4,
        "p":1024**5, "pb":1024**5, "pib":1024**5,
    }
    mult = table.get(unit)
    return int(val * mult) if mult else None

def fmt_bytes(n: Optional[int]) -> str:
    """Human-readable IEC format."""
    if not n or n <= 0:
        return "n/a"
    for unit, m in [("PiB",1024**5),("TiB",1024**4),("GiB",1024**3),("MiB",1024**2),("KiB",1024)]:
        if n >= m:
            return f"{n/m:.2f} {unit}"
    return f"{n} B"

def secs_to_h(s: Optional[int]) -> str:
    return "n/a" if s is None else f"{s/3600:.2f}"

def pct_str(x: Optional[float]) -> str:
    return f"{x*100:.2f}%" if x is not None else "n/a"

def run(cmd: List[str]) -> str:
    return subprocess.check_output(cmd, text=True, errors="ignore")

# ---------- select=... parsing ----------
def parse_total_ncpus_from_select(select_str: str) -> Optional[int]:
    total = 0
    for chunk in select_str.split("+"):
        fields = chunk.split(":")
        count, idx = (int(fields[0]), 1) if fields and fields[0].isdigit() else (1, 0)
        ncpus = 0
        for f in fields[idx:]:
            if f.startswith("ncpus="):
                try:
                    ncpus = int(f.split("=",1)[1])
                except ValueError:
                    pass
        total += count * ncpus
    return total or None

def parse_total_mem_from_select(select_str: str) -> Optional[int]:
    total, seen = 0, False
    for chunk in select_str.split("+"):
        fields = chunk.split(":")
        count, idx = (int(fields[0]), 1) if fields and fields[0].isdigit() else (1, 0)
        mem_bytes = None
        for f in fields[idx:]:
            if f.startswith("mem="):
                mem_bytes = parse_size_to_bytes(f.split("=",1)[1])
                seen = True
        if mem_bytes:
            total += count * mem_bytes
    return total if seen else None

# ---------- Single job summarization ----------
def summarize_job_from_block(block: str) -> Dict[str, Any]:
    jobid = (RE_JOBID.search(block).group(1) if RE_JOBID.search(block) else "unknown")
    name  = (RE_NAME.search(block).group(1) if RE_NAME.search(block) else "")
    state = (RE_STATE.search(block).group(1) if RE_STATE.search(block) else "?")

    nodes = None
    m_exec = RE_EXECHOST.search(block)
    if m_exec:
        hosts_raw = re.sub(r"\s+", "", m_exec.group(1))
        host_list: List[str] = []
        for chunk in hosts_raw.split("+"):
            host = chunk.split("/")[0]
            if host and host not in host_list:
                host_list.append(host)
        nodes = ",".join(host_list) if host_list else None

    cput_s = hms_to_seconds(RE_CPUT.search(block).group(1)) if RE_CPUT.search(block) else None
    wall_s = hms_to_seconds(RE_WALL.search(block).group(1)) if RE_WALL.search(block) else None

    ncpus_m = RE_NCPUS.search(block)
    ncpus = int(ncpus_m.group(1)) if ncpus_m else None
    if ncpus is None and RE_SELECT.search(block):
        ncpus = parse_total_ncpus_from_select(RE_SELECT.search(block).group(1))

    used_mem_b  = parse_size_to_bytes(RE_MEMU.search(block).group(1))  if RE_MEMU.search(block)  else None
    used_vmem_b = parse_size_to_bytes(RE_VMEMU.search(block).group(1)) if RE_VMEMU.search(block) else None

    req_mem_b = parse_size_to_bytes(RE_MEMREQ.search(block).group(1)) if RE_MEMREQ.search(block) else None
    if req_mem_b is None and RE_SELECT.search(block):
        req_mem_b = parse_total_mem_from_select(RE_SELECT.search(block).group(1))

    req_vmem_b = parse_size_to_bytes(RE_VMEMRQ.search(block).group(1)) if RE_VMEMRQ.search(block) else None

    avg_used_cpus = (cput_s / wall_s) if (cput_s is not None and wall_s and wall_s > 0) else None
    cpu_eff = (avg_used_cpus / ncpus) if (avg_used_cpus is not None and ncpus) else None
    mem_eff = (used_mem_b / req_mem_b) if (used_mem_b and req_mem_b and req_mem_b > 0) else None
    vmem_eff = (used_vmem_b / req_vmem_b) if (used_vmem_b and req_vmem_b and req_vmem_b > 0) else None

    return {
        "jobid": jobid, "name": name, "state": state, "nodes": nodes,
        "ncpus": ncpus, "wall_s": wall_s, "cput_s": cput_s,
        "avg_used_cpus": avg_used_cpus, "cpu_eff": cpu_eff,
        "used_mem_b": used_mem_b, "req_mem_b": req_mem_b, "mem_eff": mem_eff,
        "used_vmem_b": used_vmem_b, "req_vmem_b": req_vmem_b, "vmem_eff": vmem_eff,
    }

def summarize_job(jobid: str) -> Dict[str, Any]:
    blk = run(["qstat", "-fx", jobid])
    return summarize_job_from_block(blk)

# ---------- Fast bulk path ----------
def split_qstat_f_blocks(blob: str) -> List[str]:
    """Split a single 'qstat -f ...' response into per-job blocks."""
    lines = blob.splitlines()
    idxs = [i for i, ln in enumerate(lines) if ln.strip().startswith("Job Id:")]
    blocks = []
    for i, start in enumerate(idxs):
        end = idxs[i+1] if i+1 < len(idxs) else len(lines)
        blocks.append("\n".join(lines[start:end]))
    return blocks


def summarize_all_jobs_bulk(user: str, include_finished: bool) -> list[dict]:
    # Try with -u first; if server disallows, fall back to all and filter by owner.
    variants = [
        ["qstat"] + (["-x"] if include_finished else []) + ["-f", "-u", user, "-t"],
        ["qstat"] + (["-x"] if include_finished else []) + ["-f", "-t"],
    ]
    rows = []
    for args in variants:
        try:
            blob = run(args)
        except subprocess.CalledProcessError:
            continue
        blocks = split_qstat_f_blocks(blob)
        cand = []
        for b in blocks:
            m = RE_OWNER.search(b)
            if not m:
                continue
            owner = m.group(1)  # Job_Owner = <user>@host
            if owner == user:
                cand.append(summarize_job_from_block(b))
        if cand:
            rows = cand
            break
    return rows


# ---------- Compat path (per job) ----------
def list_user_jobids(user: str, include_finished: bool=False) -> list[str]:
    args = []
    if include_finished:
        args.append("-x")
    args += ["-u", user, "-t"]
    out = run(["qstat", *args])
    jobids = []
    for line in out.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("job id") or line.startswith("-"):
            continue
        tok = line.split()[0]
        # Accept: 12345 | 12345.server | 12345[1] | 12345[1].server
        if re.match(r"^\d+(?:\[[^\]]+\])?(?:\.\S+)?$", tok):
            jobids.append(tok)
    return jobids


def summarize_all_jobs_compat(user: str, include_finished: bool) -> List[Dict[str, Any]]:
    jobids = list_user_jobids(user, include_finished=include_finished)
    rows = []
    for jid in jobids:
        try:
            rows.append(summarize_job(jid))
        except subprocess.CalledProcessError:
            continue
    return rows

# ---------- Output ----------
def render_table(rows: List[Dict[str,Any]], name_max: int) -> None:
    cols = ["JOBID","STATE","NAME","NODES","NCPUS","WALL(h)","CPUT(h)","avgCPU","CPUeff","memUsed","memReq","memEff"]
    w = {c: len(c) for c in cols}
    table = []
    for r in rows:
        name = r.get("name") or ""
        if name_max > 0 and len(name) > name_max:
            name = name[:max(0,name_max-1)] + "…"
        row = {
            "JOBID":  r["jobid"],
            "STATE":  r["state"],
            "NAME":   name,
            "NODES":  r.get("nodes") or "n/a",
            "NCPUS":  str(r["ncpus"] if r["ncpus"] is not None else "n/a"),
            "WALL(h)": secs_to_h(r["wall_s"]),
            "CPUT(h)": secs_to_h(r["cput_s"]),
            "avgCPU": f"{r['avg_used_cpus']:.2f}" if r["avg_used_cpus"] is not None else "n/a",
            "CPUeff": pct_str(r["cpu_eff"]),
            "memUsed": fmt_bytes(r["used_mem_b"]),
            "memReq":  fmt_bytes(r["req_mem_b"]),
            "memEff":  pct_str(r["mem_eff"]),
        }
        for k,v in row.items():
            w[k] = max(w[k], len(str(v)))
        table.append(row)
    hdr = "  ".join(f"{c:<{w[c]}}" for c in cols)
    print(hdr)
    print("-" * len(hdr))
    for row in table:
        print("  ".join(f"{row[c]:<{w[c]}}" for c in cols))

def write_csv(rows: List[Dict[str,Any]], path: str) -> None:
    import csv
    fields = [
        "jobid","name","state","nodes","ncpus","wall_s","cput_s","avg_used_cpus","cpu_eff",
        "used_mem_b","used_mem_gb","req_mem_b","req_mem_gb","mem_eff",
        "used_vmem_b","used_vmem_gb","req_vmem_b","req_vmem_gb","vmem_eff",
    ]
    f = sys.stdout if path == "-" else open(path, "w", newline="")
    with f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            row = {
                "jobid": r.get("jobid"),
                "name": r.get("name"),
                "state": r.get("state"),
                "nodes": r.get("nodes"),
                "ncpus": r.get("ncpus"),
                "wall_s": r.get("wall_s"),
                "cput_s": r.get("cput_s"),
                "avg_used_cpus": r.get("avg_used_cpus"),
                "cpu_eff": r.get("cpu_eff"),
                "used_mem_b": r.get("used_mem_b"),
                "req_mem_b": r.get("req_mem_b"),
                "mem_eff": r.get("mem_eff"),
                "used_vmem_b": r.get("used_vmem_b"),
                "req_vmem_b": r.get("req_vmem_b"),
                "vmem_eff": r.get("vmem_eff"),
            }

            if row["avg_used_cpus"] is not None:
                row["avg_used_cpus"] = round(row["avg_used_cpus"], 2)

            for src, dest in [
                ("used_mem_b", "used_mem_gb"),
                ("req_mem_b", "req_mem_gb"),
                ("used_vmem_b", "used_vmem_gb"),
                ("req_vmem_b", "req_vmem_gb"),
            ]:
                val = row.get(src)
                row[dest] = (val / (1024**3)) if val is not None else None

            w.writerow({k: row.get(k) for k in fields})

def aggregate(rows: List[Dict[str,Any]]) -> Dict[str,float]:
    import math
    def mean(xs):
        xs = [x for x in xs if x is not None and not (isinstance(x,float) and (math.isnan(x) or math.isinf(x)))]
        return sum(xs)/len(xs) if xs else float("nan")
    return {
        "jobs": len(rows),
        "avg_CPUeff_%": mean([r["cpu_eff"]*100 for r in rows if r.get("cpu_eff") is not None]),
        "avg_avgCPU":   mean([r["avg_used_cpus"] for r in rows if r.get("avg_used_cpus") is not None]),
        "avg_memEff_%": mean([r["mem_eff"]*100 for r in rows if r.get("mem_eff") is not None]),
    }

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(
        description="PBS job CPU & Memory statistics (single job or all jobs for a user)."
    )
    ap.add_argument("--job", help="Job ID to summarize (default: $PBS_JOBID)")
    ap.add_argument("--user", help="Summarize all jobs of USER")
    ap.add_argument("--include-finished", action="store_true", help="Include finished jobs (qstat -x)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--mode", choices=["bulk","compat"], default="bulk",
    help="Bulk mode: one qstat -f for all jobs; auto-fallback to compat if it fails.\n Compat mode: one qstat -fx per job (slower but widely compatible)"
    )
#    mode.add_argument("--fast", action="store_true", help="Bulk mode: one qstat -f for all jobs; auto-fallback to compat if it fails")
#    mode.add_argument("--compat", action="store_true", help="Compat mode: one qstat -fx per job (slower but widely compatible)")
    ap.add_argument("--csv", metavar="PATH", help='Write CSV to PATH (use "-" for stdout)')
    ap.add_argument("--name-max", type=int, default=30, help="Max width for job name column; 0=disable truncation (default: 30)")
    args = ap.parse_args()

    if not shutil.which("qstat"):
        print("ERROR: qstat not found in PATH.", file=sys.stderr)
        sys.exit(1)

    rows: List[Dict[str,Any]] = []

    # single-job path
    if not args.user:
        jobid = args.job or os.environ.get("PBS_JOBID")
        if not jobid:
            print("Provide --job JOBID or run inside PBS with $PBS_JOBID set, or use --user USER.", file=sys.stderr)
            sys.exit(2)
        rows.append(summarize_job(jobid))
    else:
        # multi-job path
        used_fast = False
        if args.mode == "bulk":
            try:
                rows = summarize_all_jobs_bulk(args.user, include_finished=args.include_finished)
                used_fast = True
            except subprocess.CalledProcessError:
                rows = []
            except Exception:
                rows = []
        if not rows and args.mode == "compat":
            rows = summarize_all_jobs_compat(args.user, include_finished=args.include_finished)

    # output
    render_table(rows, name_max=args.name_max)
    agg = aggregate(rows)
    print("\nSummary:")
    print(f"  jobs:        {agg['jobs']}")
    if agg['avg_CPUeff_%'] == agg['avg_CPUeff_%']:
        print(f"  mean CPUeff: {agg['avg_CPUeff_%']:.2f}%")
    if agg['avg_avgCPU'] == agg['avg_avgCPU']:
        print(f"  mean avgCPU: {agg['avg_avgCPU']:.2f}")
    if agg['avg_memEff_%'] == agg['avg_memEff_%']:
        print(f"  mean memEff: {agg['avg_memEff_%']:.2f}%")

    if args.csv:
        write_csv(rows, args.csv)

if __name__ == "__main__":
    main()
