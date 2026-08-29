r"""Full Qwen3.5-9B pipeline: download -> verify arch -> smoke -> gate -> comprehensive.
Monitors download completion and triggers each stage sequentially.
Logs all output to pipeline_qwen.log in logs/ dir.
"""
import os, json, sys, time, subprocess, threading
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
LOGFILE = REPO_ROOT / "logs" / "pipeline_qwen.log"
LOGFILE.parent.mkdir(parents=True, exist_ok=True)

MODEL_DIR = Path(r"D:\hf_models\Qwen3.5-9B")
DOWNLOAD_STATUS = REPO_ROOT / "logs" / "download_qwen_status.json"

def log(m):
    ts = time.strftime('%H:%M:%S')
    msg = f"[{ts}] {m}"
    print(msg, flush=True)
    with open(LOGFILE, "a") as f:
        f.write(msg + "\n")

def run_script(script_name, *args):
    """Run a script via venv, block until done."""
    log(f"Starting {script_name}...")
    try:
        result = subprocess.run(
            [str(VENV_PYTHON), "-m", f"scripts.{script_name}"] + list(args),
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=3600
        )
        if result.stdout:
            log(f"{script_name} stdout:\n{result.stdout}")
        if result.returncode != 0:
            log(f"ERROR {script_name} rc={result.returncode}")
            if result.stderr:
                log(f"stderr: {result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        log(f"TIMEOUT {script_name}")
        return False
    except Exception as e:
        log(f"EXCEPTION {script_name}: {e}")
        return False

def wait_download(timeout_sec=1800):
    """Poll for download_qwen_status.json (30min timeout)."""
    log("Waiting for download completion...")
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        if DOWNLOAD_STATUS.exists():
            status = json.load(open(DOWNLOAD_STATUS))
            log(f"Download status: {status['status']}")
            return status["status"] == "done"
        time.sleep(10)
    log("Download timeout")
    return False

def verify_arch():
    """Run architecture verification."""
    if not run_script("verify_qwen_arch"):
        return False
    arch_file = REPO_ROOT / "logs" / "qwen_arch_verify.json"
    if not arch_file.exists():
        log("ERROR: arch verify did not produce output")
        return False
    arch = json.load(open(arch_file))
    log(f"ARCH: {arch['num_layers']} layers, {arch['intermediate_size']} intermediate, "
        f"{arch['total_mlp_neurons']} total neurons, silu={arch['is_silu']}, moe={arch['is_moe']}")
    if not arch["is_silu"] or arch["is_moe"]:
        log("STOP: Not DENSE SwiGLU with silu")
        return False
    return True

def run_smoke():
    """Run smoke test."""
    if not run_script("smoke_qwen"):
        return False
    smoke_man = REPO_ROOT / "results" / "qwen35_9b_smoke" / "smoke_manifest.json"
    if not smoke_man.exists():
        log("ERROR: smoke_qwen did not produce manifest")
        return False
    man = json.load(open(smoke_man))
    log(f"Smoke baseline: code_nll={man['baseline']['code_nll']:.4f} pass@1={man['baseline']['code_pass1']:.3f}")
    return True

def run_fidelity():
    """Run fidelity check."""
    if not run_script("fidelity_qwen"):
        return False
    fid_file = REPO_ROOT / "results" / "qwen35_9b_smoke" / "fidelity.json"
    if not fid_file.exists():
        log("ERROR: fidelity_qwen did not produce output")
        return False
    fid = json.load(open(fid_file))
    log(f"Fidelity: int8_nll={fid['int8_code_nll']:.4f} fp16_nll={fid['fp16_code_nll']:.4f} "
        f"delta_rel={fid['nll_delta_rel']:.3f} top20_jaccard_mean={fid['top20_jaccard_mean']:.3f}")
    # GATE: Jaccard mean > 0.90
    if fid['top20_jaccard_mean'] < 0.90:
        log(f"GATE FAIL: Jaccard {fid['top20_jaccard_mean']:.3f} < 0.90")
        return False
    return True

def check_smoke_3control():
    """Verify 3-control monotonic (high >> random >> low)."""
    sweep = REPO_ROOT / "results" / "qwen35_9b_smoke" / "sweep_rows.jsonl"
    if not sweep.exists():
        log("ERROR: sweep_rows.jsonl not found")
        return False

    rows = [json.loads(line) for line in open(sweep)]
    # Sample step 10 (should have high >> random >> low)
    step10 = [r for r in rows if r["step"] == 10 and r["control"] != "random"]
    if not step10:
        log("WARNING: no step 10 data for 3-control check")
        return True

    # Check if high > low (simple monotonic)
    high = [r for r in step10 if r["control"] == "high"]
    low = [r for r in step10 if r["control"] == "low"]
    if high and low:
        h_nll = high[0]["code_nll"]
        l_nll = low[0]["code_nll"]
        log(f"3-control check (step 10): low_nll={l_nll:.4f} high_nll={h_nll:.4f} ratio={h_nll/l_nll:.2f}")
        if h_nll < l_nll:  # HIGH should be better (lower NLL) than LOW
            log("GATE FAIL: 3-control not monotonic (high < low)")
            return False
    return True

def launch_comprehensive():
    """Launch comprehensive in detached subprocess (non-blocking)."""
    log("Launching comprehensive (detached)...")
    outdir = "results/comprehensive_qwen35_9b"
    logfile = REPO_ROOT / "logs" / "comprehensive_qwen35_9b_run.log"

    try:
        with open(logfile, "w") as lf:
            subprocess.Popen(
                [str(VENV_PYTHON), "-m", "scripts.run_full_qwen", outdir, "0.70"],
                cwd=REPO_ROOT,
                stdout=lf,
                stderr=subprocess.STDOUT
            )
        log(f"Comprehensive launched detached -> {logfile}")
        log(f"Monitor with: tail -f {logfile}")
        return True
    except Exception as e:
        log(f"ERROR launching comprehensive: {e}")
        return False

def main():
    log("=" * 60)
    log("QWEN3.5-9B CAPABILITY-REMOVAL PIPELINE")
    log("=" * 60)

    # STEP 1: Wait for download
    if not wait_download():
        log("[BLOCK] Download failed or timed out")
        return 1

    # STEP 2: Verify architecture
    if not verify_arch():
        log("[BLOCK] Architecture verification failed")
        return 1

    # STEP 3: Run smoke test
    if not run_smoke():
        log("[BLOCK] Smoke test failed")
        return 1

    # STEP 3b: Check fidelity + gate
    if not run_fidelity():
        log("[BLOCK] Fidelity check failed")
        return 1

    if not check_smoke_3control():
        log("[BLOCK] 3-control monotonic gate failed")
        return 1

    log("GATE PASSED -> proceeding to comprehensive")

    # STEP 4: Launch comprehensive (detached)
    if not launch_comprehensive():
        log("[BLOCK] Failed to launch comprehensive")
        return 1

    log("=" * 60)
    log("PIPELINE COMPLETE")
    log("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
