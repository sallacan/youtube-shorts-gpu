"""RunPod Serverless handler for YouTube Shorts generation."""
import runpod
import traceback
import torch

# Log GPU info at worker startup so we can diagnose CUDA issues
if torch.cuda.is_available():
    print(f"[GPU] Device: {torch.cuda.get_device_name(0)}")
    print(f"[GPU] Compute capability: {torch.cuda.get_device_capability(0)}")
    print(f"[GPU] VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
else:
    print("[GPU] No CUDA device available")


def handler(job):
    try:
        # Lazy import - only runs when GPU is available
        from app import run_job
        result = run_job(job["input"])
        return result
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
