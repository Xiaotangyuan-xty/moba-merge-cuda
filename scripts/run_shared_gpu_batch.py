#!/usr/bin/env python3
"""Approved coverage batch only: physical GPU1, <=6 GiB, <=15 min cumulative.

Monitors via nvidia-smi. If another process appears, terminate only the child
process group created here. No process discovery/kill outside that group.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import hashlib

GPU_UUID=os.environ.get("MOBA_COVERAGE_GPU_UUID")
ROOT=Path(__file__).resolve().parents[1]
LIMIT_SECONDS=900
LIMIT_MIB=6144


def inspect():
    if not GPU_UUID:
        raise RuntimeError("Set MOBA_COVERAGE_GPU_UUID in private env.local.sh after approval")
    gpu_text=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,name,memory.used,memory.total",
                                      "--format=csv,noheader,nounits"],text=True,timeout=10)
    proc_text=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                                       "--format=csv,noheader,nounits"],text=True,timeout=10)
    devices=[list(map(str.strip,row)) for row in csv.reader(gpu_text.splitlines())]
    target=next((row for row in devices if row[1]==GPU_UUID),None)
    if target is None or target[0]!="1":
        raise RuntimeError("Approved physical GPU1 UUID is absent or changed; refusing to run")
    procs=[list(map(str.strip,row)) for row in csv.reader(proc_text.splitlines()) if row]
    pids={int(row[1]) for row in procs if row[0]==GPU_UUID}
    return int(target[3]),pids,{"gpus":gpu_text,"processes":proc_text}


def terminate_owned(proc):
    if proc.poll() is None:
        # start_new_session=True below guarantees this PGID is our child only.
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL)
            proc.wait()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tag",required=True)
    ap.add_argument("command",nargs=argparse.REMAINDER)
    args=ap.parse_args()
    cmd=args.command[1:] if args.command[:1]==["--"] else args.command
    if not cmd:raise SystemExit("No command")
    out=ROOT/"results"/"coverage_20261101"
    out.mkdir(parents=True,exist_ok=True)
    state_path=out/"gpu_budget.json"
    state=json.loads(state_path.read_text()) if state_path.exists() else {"used_seconds":0,"runs":[]}
    remaining=LIMIT_SECONDS-state["used_seconds"]
    if remaining<=0:raise SystemExit("Approved 15-minute GPU batch budget exhausted")
    used,pids,before=inspect()
    if pids or used>64:
        (out/(args.tag+"_blocked.json")).write_text(json.dumps(before,indent=2))
        raise SystemExit("GPU1 is occupied; no task started and no other GPU selected")
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=GPU_UUID,CUDA_DEVICE_ORDER="PCI_BUS_ID",
               OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MAX_JOBS="2")
    log_path=ROOT/"logs"/(args.tag+".log")
    log_path.parent.mkdir(exist_ok=True)
    started=time.monotonic()
    source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                   for folder in ("src","tests","reference","scripts","benchmarks","configs")
                   for p in (ROOT/folder).rglob("*") if p.is_file() and p.suffix in (".py",".cu",".cpp",".sh",".yaml")}
    reason="completed";peak=0;proc=None
    with log_path.open("w") as logfile:
        try:
            proc=subprocess.Popen(["nice","-n","10",*cmd],cwd=ROOT,env=env,
                                  stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True)
            while proc.poll() is None:
                time.sleep(1)
                used,pids,_=inspect()
                peak=max(peak,used)
                if pids-{proc.pid}:
                    reason="foreign process entered GPU1";break
                if used>LIMIT_MIB:
                    reason="6 GiB batch memory ceiling exceeded";break
                if time.monotonic()-started>remaining:
                    reason="15-minute cumulative batch time exhausted";break
        except BaseException as exc:
            reason=f"monitor/runner error: {exc}"
            raise
        finally:
            if proc is not None:
                terminate_owned(proc)
            elapsed=time.monotonic()-started
            run={"tag":args.tag,"command":cmd,"gpu_uuid":GPU_UUID,"wall_seconds":elapsed,
                 "peak_device_mib":peak,"reason":reason,"returncode":None if proc is None else proc.returncode,
                 "before":before,"source_hashes":source_hashes,
                 "git_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()}
            state["used_seconds"]+=elapsed;state["runs"].append(run)
            state_path.write_text(json.dumps(state,indent=2))
    print(json.dumps({k:v for k,v in run.items() if k not in ("before","source_hashes")}))
    raise SystemExit(0 if reason=="completed" and proc.returncode==0 else 1)


if __name__=="__main__":main()
