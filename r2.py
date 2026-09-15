"""Upload finished renders to Cloudflare R2 (S3-compatible) using SigV4 presigned URLs.

No SDK on purpose: the signing is a few lines of hashlib/hmac, and it keeps boto3 and its
dependency tree out of an image whose pins have crash-looped the worker before.

Credentials come from environment variables that the RunPod template fills from encrypted
RunPod Secrets ({{ RUNPOD_SECRET_... }}), so the key never appears in the template itself.
"""
import datetime
import hashlib
import hmac
import os
import subprocess
import tempfile
import urllib.parse

GET_LINK_SECONDS = 3 * 24 * 3600   # Kontrol Memuru downloads within the hour; 3 days covers manual recovery.
                                   # The bucket's lifecycle rule deletes the object after 7 days anyway.


def _hmac(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode(), hashlib.sha256).digest()


def presign(method, host, path, access_key, secret_key, expires, region="auto", service="s3", now=None):
    """AWS Signature V4 query-string auth, signing only the Host header (UNSIGNED-PAYLOAD)."""
    now = now or datetime.datetime.now(datetime.timezone.utc)
    amz_date, day = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
    scope = f"{day}/{region}/{service}/aws4_request"
    params = {
        "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
        "X-Amz-Credential": f"{access_key}/{scope}",
        "X-Amz-Date": amz_date,
        "X-Amz-Expires": str(int(expires)),
        "X-Amz-SignedHeaders": "host",
    }
    uri = urllib.parse.quote(path, safe="/~")
    query = "&".join(f"{urllib.parse.quote(k, safe='~')}={urllib.parse.quote(v, safe='~')}"
                     for k, v in sorted(params.items()))
    canonical = "\n".join([method, uri, query, f"host:{host}\n", "host", "UNSIGNED-PAYLOAD"])
    to_sign = "\n".join(["AWS4-HMAC-SHA256", amz_date, scope,
                         hashlib.sha256(canonical.encode()).hexdigest()])
    k = _hmac(("AWS4" + secret_key).encode(), day)
    for part in (region, service, "aws4_request"):
        k = _hmac(k, part)
    signature = hmac.new(k, to_sign.encode(), hashlib.sha256).hexdigest()
    return f"https://{host}{uri}?{query}&X-Amz-Signature={signature}"


def config():
    """(host, bucket, access_key, secret_key) or None, plus a reason when unusable."""
    acct = os.environ.get("R2_ACCOUNT_ID", "").strip()
    bucket = os.environ.get("R2_BUCKET", "").strip()
    ak = os.environ.get("R2_ACCESS_KEY_ID", "").strip()
    sk = os.environ.get("R2_SECRET_ACCESS_KEY", "").strip()
    if not (acct and bucket and ak and sk):
        return None, "r2: not configured"
    if "RUNPOD_SECRET" in ak or "RUNPOD_SECRET" in sk or "{{" in ak or "{{" in sk:
        return None, "r2: secret reference was not substituted by RunPod"
    return (f"{acct}.r2.cloudflarestorage.com", bucket, ak, sk), None


def upload_file(local_path, key, content_type="video/mp4", timeout=300):
    """PUT the file to R2 and return (download_url, None) or (None, error)."""
    cfg, why = config()
    if not cfg:
        return None, why
    host, bucket, ak, sk = cfg
    path = f"/{bucket}/{key}"
    put_url = presign("PUT", host, path, ak, sk, 900)
    with tempfile.NamedTemporaryFile(suffix=".txt") as body:
        r = subprocess.run(
            ["curl", "-s", "--max-time", str(timeout), "-o", body.name, "-w", "%{http_code}",
             "-X", "PUT", "-H", f"Content-Type: {content_type}", "--upload-file", local_path, put_url],
            capture_output=True, text=True)
        code = (r.stdout or "").strip()[-3:]
        if r.returncode != 0 or code != "200":
            detail = open(body.name, errors="replace").read()[:160]
            return None, f"r2: rc={r.returncode} http={code} {detail}"
    return presign("GET", host, path, ak, sk, GET_LINK_SECONDS), None


def self_test():
    """Tiny PUT + GET round trip for the diagnostic job. Never returns key material."""
    cfg, why = config()
    out = {"configured": bool(cfg), "reason": why,
           "access_key_len": len(os.environ.get("R2_ACCESS_KEY_ID", "")),
           "secret_len": len(os.environ.get("R2_SECRET_ACCESS_KEY", ""))}
    if not cfg:
        return out
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("r2 self-test\n")
    try:
        url, err = upload_file(f.name, "diag/self-test.txt", content_type="text/plain", timeout=60)
        out["put"] = "ok" if url else err
        if url:
            g = subprocess.run(["curl", "-s", "--max-time", "30", "-o", "/dev/null", "-w", "%{http_code}", url],
                               capture_output=True, text=True)
            out["get_http"] = (g.stdout or "").strip()
    finally:
        os.unlink(f.name)
    return out
