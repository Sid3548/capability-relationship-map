"""Download Qwen3.5-9B to D:\hf_models\Qwen3.5-9B (safetensors+config+tokenizer only)."""
import os, sys, json, time
from pathlib import Path
from huggingface_hub import snapshot_download

HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_ID = "Qwen/Qwen3.5-9B"
LOCAL_DIR = r"D:\hf_models\Qwen3.5-9B"
LOGFILE = Path(__file__).parent.parent / "logs" / "download_qwen_log.txt"
LOGFILE.parent.mkdir(parents=True, exist_ok=True)

def log(m):
    ts = time.strftime('%H:%M:%S')
    msg = f"[{ts}] {m}"
    print(msg, flush=True)
    with open(LOGFILE, "a") as f:
        f.write(msg + "\n")

try:
    log(f"Starting download: {MODEL_ID} -> {LOCAL_DIR}")
    Path(LOCAL_DIR).mkdir(parents=True, exist_ok=True)
    snapshot_download(
        MODEL_ID,
        cache_dir=r"D:\hf_models",
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,
        allow_patterns=["*.safetensors", "config.json", "*.tokenizer.json", "tokenizer.model", "*.txt"],
        token=HF_TOKEN,
    )
    log(f"Download complete -> {LOCAL_DIR}")
    with open(LOGFILE.parent / "download_qwen_status.json", "w") as f:
        json.dump({"status": "done", "local_dir": LOCAL_DIR, "timestamp": time.time()}, f)
except Exception as e:
    log(f"ERROR: {e}")
    with open(LOGFILE.parent / "download_qwen_status.json", "w") as f:
        json.dump({"status": "error", "error": str(e), "timestamp": time.time()}, f)
    sys.exit(1)
