#!/usr/bin/env python
r"""Monitor Qwen3.5-9B pipeline progress. Run this to check status."""
import json, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS = REPO_ROOT / "logs"

def get_size_gb(path):
    """Get total dir size in GB."""
    if not path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
    return total / (1024**3)

def main():
    print("\n" + "="*60)
    print("QWEN3.5-9B PIPELINE STATUS")
    print("="*60)

    # Download status
    download_log = LOGS / "download_qwen_log.txt"
    download_status = LOGS / "download_qwen_status.json"
    model_dir = Path(r"D:\hf_models\Qwen3.5-9B")

    print("\n[DOWNLOAD]")
    if download_log.exists():
        print("  Last line:", open(download_log).readlines()[-1].strip())
    if download_status.exists():
        status = json.load(open(download_status))
        print(f"  Status: {status['status']}")
    else:
        size = get_size_gb(model_dir)
        print(f"  Status: IN PROGRESS (~{size:.1f} GB so far)")

    # Pipeline status
    print("\n[PIPELINE ORCHESTRATOR]")
    pipeline_log = LOGS / "pipeline_qwen.log"
    if pipeline_log.exists():
        lines = open(pipeline_log).readlines()
        for line in lines[-10:]:
            print(" ", line.rstrip())
    else:
        print("  (not yet started)")

    # Result directories
    print("\n[RESULTS]")
    smoke_dir = REPO_ROOT / "results" / "qwen35_9b_smoke"
    comp_dir = REPO_ROOT / "results" / "comprehensive_qwen35_9b"

    if smoke_dir.exists() and (smoke_dir / "smoke_manifest.json").exists():
        print("  ✓ smoke_qwen.py completed")
        man = json.load(open(smoke_dir / "smoke_manifest.json"))
        print(f"    - baseline code_nll: {man['baseline']['code_nll']:.4f}")
        print(f"    - baseline pass@1: {man['baseline']['code_pass1']:.3f}")

    if (smoke_dir / "fidelity.json").exists():
        fid = json.load(open(smoke_dir / "fidelity.json"))
        print(f"  ✓ fidelity_qwen.py completed")
        print(f"    - jaccard_mean: {fid['top20_jaccard_mean']:.3f} {'(GATE PASS)' if fid['top20_jaccard_mean'] >= 0.90 else '(GATE FAIL)'}")

    if comp_dir.exists() and (comp_dir / "manifest.json").exists():
        print("  ✓ run_full_qwen.py in progress or complete")
        if (comp_dir / "A_overlap.json").exists():
            print("    - A_overlap.json done")
        if (comp_dir / "B_allocation.json").exists():
            print("    - B_allocation.json done")
        if (comp_dir / "C_stress.json").exists():
            print("    - C_stress.json done")
        if (comp_dir / "D_interlink.json").exists():
            print("    - D_interlink.json done")

    # Comprehensive log
    comp_log = LOGS / "comprehensive_qwen35_9b_run.log"
    if comp_log.exists():
        lines = open(comp_log).readlines()
        if lines:
            print(f"  Comprehensive log tail (last 5 lines):")
            for line in lines[-5:]:
                print(f"    {line.rstrip()}")

    print("\n" + "="*60)
    print("Logs:")
    print(f"  Download:   {LOGS / 'download_qwen_log.txt'}")
    print(f"  Pipeline:   {LOGS / 'pipeline_qwen.log'}")
    print(f"  Comprehensive: {comp_log}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
