"""RunPod Serverless handler for YouTube Shorts generation."""
import runpod
import traceback


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
