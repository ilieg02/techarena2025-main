#!/usr/bin/env python3
import os, subprocess, sys, time, json


PYTHON=sys.executable
SCRIPTS = {
    "data_prep": "data_prep.py",
    "train_biencoder": "train_biencoder.py",
    "train_reranker": "train_reranker.py",
    "inference": "inference_rag.py"
}
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def run_step(name, script, env=None):
    print(f"\n=== RUNNING: {name} ===")
    start = time.time()
    cmd = [PYTHON, script]
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log_path = os.path.join(LOG_DIR, f"{name}.log")
    with open(log_path, "w", encoding="utf-8") as f:
        for line in p.stdout:
            print(line, end="")
            f.write(line)
    p.wait()
    duration = time.time() - start
    print(f"=== {name} finished (exit {p.returncode}) in {duration:.1f}s; log: {log_path} ===")
    if p.returncode != 0:
        raise SystemExit(f"Step {name} failed (see {log_path})")

def check_exists(files):
    for f in files:
        if not os.path.exists(f):
            return False
    return True

def main():
    # 1) Data prep
    run_step("data_prep", SCRIPTS["data_prep"])

    # 2) Train biencoder (skip if trained model exists)
    if not os.path.exists("biencoder/model"):
        run_step("train_biencoder", SCRIPTS["train_biencoder"])
    else:
        print("biencoder model exists skipping biencoder training.")

    # 3) Train reranker (skip if exists)
    if not os.path.exists("reranker"):
        run_step("train_reranker", SCRIPTS["train_reranker"])
    else:
        print("reranker exists  skipping reranker training.")

    # 4) Inference / index build and sample run
    run_step("inference", SCRIPTS["inference"])

    print("All steps completed successfully.")

if __name__ == "__main__":
    main()
