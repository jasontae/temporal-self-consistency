"""Thrashing watchdog for the LLaMA-3 seed queue (Edward's rule, 2026-09-29).

Every 60 s, while a run_tcl_mlx training process is alive and older than 10 min:
  - read its latest "fixed step N ... (Ts)" line from llama3_seeds.log;
  - keep (wall clock, step) samples for that pid;
  - PAUSE (SIGSTOP the training process and the queue) if, over the trailing
    10 minutes of wall clock, the mean step time exceeds 1.8 s (3x the 0.6 s
    baseline), or no step advanced at all;
  - PAUSE if kernel memory pressure is critical (level 4).
Never pauses on swap level. Exits when the queue logs ALL LLAMA3 SEEDS DONE.
"""
import re
import subprocess
import time
from pathlib import Path

LOG = Path(__file__).with_name("llama3_seeds.log")
WINDOW, LIMIT, BASE_AGE = 600, 1.8, 600
STEP = re.compile(r"fixed step (\d+) ")


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def note(msg):
    with open(LOG, "a") as f:
        f.write(f"=== watchdog {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}: {msg}\n")


def pause(reason):
    pids = sh("pgrep -f 'src.training.run_tcl_mlx|run_llama3_queue.sh|run_llama3_seed.sh'").split()
    for p in pids:
        subprocess.run(["kill", "-STOP", p])
    note(f"PAUSED ({reason}); SIGSTOP {' '.join(pids)}; resume with: kill -CONT {' '.join(pids)}")


note("started (pause only on critical pressure or step time > 1.8 s over 10 min)")
samples, last_pid = [], None
while "ALL LLAMA3 SEEDS DONE" not in LOG.read_text():
    time.sleep(60)
    if sh("sysctl -n kern.memorystatus_vm_pressure_level") == "4":
        pause("memory pressure critical"); break
    pid = sh("pgrep -f 'src.training.run_tcl_mlx'").split()
    if not pid:
        samples, last_pid = [], None
        continue
    pid = pid[0]
    if pid != last_pid:
        samples, last_pid = [], pid
    et = sh(f"ps -o etimes= -p {pid}")
    if not et.isdigit():
        continue
    steps = STEP.findall(LOG.read_text()[-20000:])
    step = int(steps[-1]) if steps else 0
    now = time.time()
    samples.append((now, step))
    samples = [s for s in samples if now - s[0] <= WINDOW + 90]
    if int(et) < BASE_AGE or now - samples[0][0] < WINDOW:
        continue
    d_steps = step - samples[0][1]
    per_step = (now - samples[0][0]) / d_steps if d_steps > 0 else float("inf")
    if per_step > LIMIT:
        pause(f"step time {per_step:.2f} s over the last {int(now - samples[0][0])} s (> {LIMIT} s)"); break
note("exiting")
