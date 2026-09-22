# Derived from igorls/ninfer @ 5e4a66d (Apache-2.0), modified for this tree (ninfer-fusion).
import argparse
import json
import os
import sys
from typing import Dict, List, Tuple
import numpy as np

def load_tensor(base_dir: str, rec: dict) -> np.ndarray:
    file_path = os.path.join(base_dir, rec["file"])
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Tensor file not found: {file_path}")
        
    dtype_str = rec.get("dtype", "BF16")
    shape = rec.get("shape", [])
    
    with open(file_path, "rb") as f:
        data = f.read()
        
    if dtype_str == "BF16":
        # Convert BF16 to float32
        u16 = np.frombuffer(data, dtype=np.uint16)
        u32 = u16.astype(np.uint32) << 16
        arr = u32.view(np.float32)
    elif dtype_str == "FP32":
        arr = np.frombuffer(data, dtype=np.float32)
    elif dtype_str == "I32":
        arr = np.frombuffer(data, dtype=np.int32)
    elif dtype_str == "I64":
        arr = np.frombuffer(data, dtype=np.int64)
    else:
        arr = np.frombuffer(data, dtype=np.float32)
        
    if shape:
        try:
            arr = arr.reshape(shape)
        except Exception:
            pass
    return arr.astype(np.float64)

def compare_tensors(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    a_flat = a.flatten()
    b_flat = b.flatten()
    
    min_len = min(len(a_flat), len(b_flat))
    if len(a_flat) != len(b_flat):
        a_flat = a_flat[:min_len]
        b_flat = b_flat[:min_len]
        
    diff = np.abs(a_flat - b_flat)
    max_diff = float(np.max(diff)) if len(diff) > 0 else 0.0
    
    norm_b = float(np.linalg.norm(b_flat))
    norm_diff = float(np.linalg.norm(diff))
    rel_l2 = (norm_diff / norm_b) if norm_b > 1e-12 else norm_diff
    
    norm_a = float(np.linalg.norm(a_flat))
    if norm_a > 1e-12 and norm_b > 1e-12:
        cosine = float(np.dot(a_flat, b_flat) / (norm_a * norm_b))
    else:
        cosine = 1.0 if norm_a == norm_b else 0.0
        
    return max_diff, rel_l2, cosine

def main():
    parser = argparse.ArgumentParser(description="Compare NInfer state dumps against Oracle reference state dumps")
    parser.add_argument("ninfer_dir", help="Directory containing NInfer dump and manifest.json")
    parser.add_argument("oracle_dir", help="Directory containing Oracle dump and manifest.json")
    parser.add_argument("--threshold", type=float, default=0.05, help="Relative L2 divergence threshold (default: 0.05)")
    args = parser.parse_args()

    manifest_a_path = os.path.join(args.ninfer_dir, "manifest.json")
    manifest_b_path = os.path.join(args.oracle_dir, "manifest.json")

    if not os.path.exists(manifest_a_path):
        print(f"Error: NInfer manifest not found at {manifest_a_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(manifest_b_path):
        print(f"Error: Oracle manifest not found at {manifest_b_path}", file=sys.stderr)
        sys.exit(1)

    with open(manifest_a_path, "r", encoding="utf-8") as f:
        man_a = json.load(f)
    with open(manifest_b_path, "r", encoding="utf-8") as f:
        man_b = json.load(f)

    pos_a = {p["position"]: p for p in man_a.get("positions", [])}
    pos_b = {p["position"]: p for p in man_b.get("positions", [])}

    all_positions = sorted(set(pos_a.keys()) & set(pos_b.keys()))
    if not all_positions:
        print("Error: No overlapping positions found between dumps!", file=sys.stderr)
        sys.exit(1)

    print(f"Comparing {len(all_positions)} positions (Threshold rel-L2: {args.threshold}) ...\n")

    first_divergence = None

    for pos in all_positions:
        p_a = pos_a[pos]
        p_b = pos_b[pos]
        tok_a = p_a.get("token_id", 0)
        tok_b = p_b.get("token_id", 0)

        print(f"==================================================")
        print(f" Position {pos:04d} (Token ID: NInfer={tok_a}, Oracle={tok_b})")
        print(f"==================================================")
        print(f"{'Stage Name':<32} {'Max |d|':<12} {'Rel-L2':<12} {'Cosine':<10} {'Status'}")
        print("-" * 75)

        tensors_a = {t["name"]: t for t in p_a.get("tensors", [])}
        tensors_b = {t["name"]: t for t in p_b.get("tensors", [])}

        common_names = [t["name"] for t in p_a.get("tensors", []) if t["name"] in tensors_b]

        for name in common_names:
            rec_a = tensors_a[name]
            rec_b = tensors_b[name]

            arr_a = load_tensor(args.ninfer_dir, rec_a)
            arr_b = load_tensor(args.oracle_dir, rec_b)

            max_d, rel_l2, cos = compare_tensors(arr_a, arr_b)
            flag = "OK"
            if rel_l2 > args.threshold:
                flag = "DIVERGED"
                if first_divergence is None:
                    first_divergence = (pos, name, rel_l2, max_d, cos)

            print(f"{name:<32} {max_d:<12.5f} {rel_l2:<12.5f} {cos:<10.6f} {flag}")

            if name == "logits":
                # Compute argmax and top-5
                arr_a = np.asarray(arr_a, dtype=np.float64).reshape(-1)
                arr_b = np.asarray(arr_b, dtype=np.float64).reshape(-1)
                top5_a_idx = np.argsort(arr_a)[-5:][::-1]
                top5_b_idx = np.argsort(arr_b)[-5:][::-1]

                print("\n  --- Logits Summary ---")
                print(f"  NInfer Argmax: {top5_a_idx[0]} (logit: {arr_a[top5_a_idx[0]]:.2f})")
                print(f"  Oracle Argmax: {top5_b_idx[0]} (logit: {arr_b[top5_b_idx[0]]:.2f})")
                print("  NInfer Top-5: ", ", ".join(f"{idx}:{arr_a[idx]:.2f}" for idx in top5_a_idx))
                print("  Oracle Top-5: ", ", ".join(f"{idx}:{arr_b[idx]:.2f}" for idx in top5_b_idx))
                print()

    print("=" * 75)
    if first_divergence:
        pos, name, rel_l2, max_d, cos = first_divergence
        print(f"\n[!] FIRST DIVERGENCE DETECTED:")
        print(f"    Position: {pos}")
        print(f"    Stage:    {name}")
        print(f"    Rel-L2:   {rel_l2:.6f}")
        print(f"    Max |d|:  {max_d:.6f}")
        print(f"    Cosine:   {cos:.6f}")
    else:
        print("\n[+] ALL STAGES MATCH ORACLE WITHIN THRESHOLD!")

if __name__ == "__main__":
    main()
