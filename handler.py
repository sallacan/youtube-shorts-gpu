"""RunPod Serverless handler for YouTube Shorts generation."""
import runpod
from app import run_job


def handler(job):
    try:
        result = run_job(job["input"])
        return result
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
