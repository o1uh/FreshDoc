import time
import json
import os
import ast
import joblib
import pandas as pd
import numpy as np
import torch
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path("FreshDoc").resolve()
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from src.registry.replay import M0Scorer, M2Scorer
from src.models.baselines import extract_features, predict_proba_m1

def run_benchmark():
    gpu_available = torch.cuda.is_available()
    device_gpu = "cuda:0" if gpu_available else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "None"
    
    ckpt_dir = PROJECT_ROOT / "checkpoints" / "slot4_full" / "best"
    m1_path = PROJECT_ROOT / "checkpoints" / "m1_hgb.joblib"
    
    print(f"Loading M2Scorer from {ckpt_dir} on GPU ({gpu_name}) and CPU...")
    m2_gpu = M2Scorer(checkpoint_dir=ckpt_dir, device=device_gpu)
    m2_cpu = M2Scorer(checkpoint_dir=ckpt_dir, device="cpu")
    
    print(f"Loading M1 model from {m1_path}...")
    import sklearn._loss.loss
    sys.modules['_loss'] = sklearn._loss.loss
    m1_dict = joblib.load(m1_path)
    m1_model = m1_dict["model"] if isinstance(m1_dict, dict) and "model" in m1_dict else m1_dict
    m0 = M0Scorer()
    
    # Representative sample inputs
    sample_sig = "def process_data(items: list[str], timeout: int = 30) -> dict:"
    sample_diff = """@@ -1,4 +1,5 @@
-def process_data(items: list[str], timeout: int = 30) -> dict:
+def process_data(items: list[str], timeout: int = 60, retry: bool = True) -> dict:
+    logger.info("Processing items")
     return {}"""
    sample_doc = "Process incoming list of string items with a configurable timeout."
    code_old = "def process_data(items: list[str], timeout: int = 30) -> dict:\n    return {}"
    code_new = "def process_data(items: list[str], timeout: int = 60, retry: bool = True) -> dict:\n    logger.info('Processing items')\n    return {}"
    ast_old = ast.parse(code_old).body[0]
    ast_new = ast.parse(code_new).body[0]
    
    df_sample = pd.DataFrame([{
        "sig_old": sample_sig,
        "diff_code": sample_diff,
        "doc_old": sample_doc,
        "code_old": code_old,
        "code_new": code_new,
        "sig_type": "param_added",
    }])

    # 1. Benchmark M0 (Rule baseline)
    print("Benchmarking M0...")
    for _ in range(20):
        m0.score(sample_sig, sample_diff, sample_doc, ast_old, ast_new)
    m0_times = []
    for _ in range(200):
        t0 = time.perf_counter()
        m0.score(sample_sig, sample_diff, sample_doc, ast_old, ast_new)
        m0_times.append((time.perf_counter() - t0) * 1000.0)

    # 2. Benchmark M1 (Features + GBDT)
    print("Benchmarking M1...")
    for _ in range(10):
        feats = extract_features(df_sample)
        predict_proba_m1(m1_model, feats)
    m1_times = []
    for _ in range(100):
        t0 = time.perf_counter()
        feats = extract_features(df_sample)
        predict_proba_m1(m1_model, feats)
        m1_times.append((time.perf_counter() - t0) * 1000.0)

    # 3. Benchmark M2 GPU (Single sample)
    print("Benchmarking M2 GPU single sample...")
    for _ in range(5):
        m2_gpu.score(sample_sig, sample_diff, sample_doc, ast_old, ast_new)
    if gpu_available:
        torch.cuda.synchronize()
        
    m2_gpu_single_times = []
    for _ in range(50):
        t0 = time.perf_counter()
        m2_gpu.score(sample_sig, sample_diff, sample_doc, ast_old, ast_new)
        if gpu_available:
            torch.cuda.synchronize()
        m2_gpu_single_times.append((time.perf_counter() - t0) * 1000.0)

    # 4. Benchmark M2 GPU (Batch 16 per sample)
    print("Benchmarking M2 GPU batch 16...")
    from src.models.m2_data import encode_pair
    single_enc = encode_pair(m2_gpu.tokenizer, sample_sig, sample_diff, sample_doc, max_len=1024)
    batch_input_ids = torch.tensor([single_enc["input_ids"]] * 16, dtype=torch.long, device=m2_gpu.device)
    batch_attention_mask = torch.tensor([single_enc["attention_mask"]] * 16, dtype=torch.long, device=m2_gpu.device)
    
    with torch.no_grad():
        for _ in range(5):
            m2_gpu.model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
        if gpu_available:
            torch.cuda.synchronize()
            
        m2_gpu_batch_times = []
        for _ in range(20):
            t0 = time.perf_counter()
            m2_gpu.model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            if gpu_available:
                torch.cuda.synchronize()
            m2_gpu_batch_times.append(((time.perf_counter() - t0) * 1000.0) / 16.0)

    # 5. Benchmark M2 CPU (Single sample)
    print("Benchmarking M2 CPU single sample...")
    for _ in range(2):
        m2_cpu.score(sample_sig, sample_diff, sample_doc, ast_old, ast_new)
        
    m2_cpu_single_times = []
    for _ in range(15):
        t0 = time.perf_counter()
        m2_cpu.score(sample_sig, sample_diff, sample_doc, ast_old, ast_new)
        m2_cpu_single_times.append((time.perf_counter() - t0) * 1000.0)

    results = {
        "hardware": {
            "gpu_name": gpu_name,
            "cpu_threads": torch.get_num_threads(),
            "torch_version": torch.__version__
        },
        "m0_heuristic_cpu": {
            "mean_ms": round(float(np.mean(m0_times)), 4),
            "p50_ms": round(float(np.median(m0_times)), 4),
            "p95_ms": round(float(np.percentile(m0_times, 95)), 4),
            "samples": len(m0_times)
        },
        "m1_features_gbdt_cpu": {
            "mean_ms": round(float(np.mean(m1_times)), 4),
            "p50_ms": round(float(np.median(m1_times)), 4),
            "p95_ms": round(float(np.percentile(m1_times, 95)), 4),
            "samples": len(m1_times)
        },
        "m2_gpu_single": {
            "mean_ms": round(float(np.mean(m2_gpu_single_times)), 3),
            "p50_ms": round(float(np.median(m2_gpu_single_times)), 3),
            "p95_ms": round(float(np.percentile(m2_gpu_single_times, 95)), 3),
            "samples": len(m2_gpu_single_times)
        },
        "m2_gpu_batch16_per_item": {
            "mean_ms": round(float(np.mean(m2_gpu_batch_times)), 3),
            "p50_ms": round(float(np.median(m2_gpu_batch_times)), 3),
            "p95_ms": round(float(np.percentile(m2_gpu_batch_times, 95)), 3),
            "samples": len(m2_gpu_batch_times) * 16
        },
        "m2_cpu_single": {
            "mean_ms": round(float(np.mean(m2_cpu_single_times)), 3),
            "p50_ms": round(float(np.median(m2_cpu_single_times)), 3),
            "p95_ms": round(float(np.percentile(m2_cpu_single_times, 95)), 3),
            "samples": len(m2_cpu_single_times)
        }
    }

    out_file = PROJECT_ROOT / "reports" / "latency_profile.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print("\nBenchmark successfully finished!")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    run_benchmark()
