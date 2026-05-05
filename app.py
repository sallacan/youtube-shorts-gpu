import os
import re
import uuid
import subprocess
import tempfile
import numpy as np
import soundfile as sf
import random
import torch
import torchaudio

from PIL import Image
from diffusers import StableDiffusionXLPipeline
import torch
from kokoro import KPipeline
from faster_whisper import WhisperModel


# ── Global model cache ──────────────────────────────────────────────
_sdxl_pipe = None
_tts_pipe = None
_whisper_model = None

MUSIC_DIR = "/workspace/music"
FONTS_DIR = "/workspace/fonts"
OUTPUT_DIR = "/workspace/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_sdxl():
    global _sdxl_pipe
    if _sdxl_pipe is None:
        print("[MODEL] Loading SDXL...")
        _sdxl_pipe = StableDiffusionXLPipeline.from_pretrained(
            "SG161222/RealVisXL_V4.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to("cuda")
        try:
            _sdxl_pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass  # xformers optional
    return _sdxl_pipe


def get_tts():
    global _tts_pipe
    if _tts_pipe is None:
        print("[MODEL] Loading Kokoro TTS...")
        _tts_pipe = KPipeline(lang_code="a")
    return _tts_pipe


def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        print("[MODEL] Loading Faster-Whisper...")
        _whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    return _whisper_model


# ── Helpers ─────────────────────────────────────────────────────────

def generate_image(prompt: str, pipe, width=720, height=1280) -> np.ndarray:
    result = pipe(
        prompt=prompt,
        negative_prompt="blurry, low quality, distorted, watermark, text, logo",
        width=width,
        height=height,
        num_inference_steps=30,
        guidance_scale=7.5,
    )
    return np.array(result.images[0])


def ken_burns(frame_array: np.ndarray, total_frames: int,
              out_w: int = 720, out_h: int = 1280,
              zoom_start: float = 1.0, zoom_end: float = 1.12,
              pan_x: float = 0.0, pan_y: float = 0.0) -> list:
    """
    Sub-pixel accurate Ken Burns via PIL affine transform.
    frame_array should be LARGER than out_w x out_h (e.g. 1080x1920)
    to provide working room for pan movement.
    """
    src_img = Image.fromarray(frame_array)
    h, w = frame_array.shape[:2]
    frames = []

    for i in range(total_frames):
        t = i / max(total_frames - 1, 1)
        scale = zoom_start + (zoom_end - zoom_start) * t

        # How many source pixels map to one output pixel at this zoom
        # crop region size in source coords
        crop_w = out_w / scale
        crop_h = out_h / scale

        # Floating-point center of crop in source image
        cx = w / 2.0 + pan_x * w * t
        cy = h / 2.0 + pan_y * h * t

        # Clamp: keep crop fully inside source
        cx = max(crop_w / 2.0, min(w - crop_w / 2.0, cx))
        cy = max(crop_h / 2.0, min(h - crop_h / 2.0, cy))

        # Top-left of crop in source coords (sub-pixel)
        x0 = cx - crop_w / 2.0
        y0 = cy - crop_h / 2.0

        # PIL AFFINE: output(px,py) → source(a*px + c, e*py + f)
        pil_img = src_img.transform(
            (out_w, out_h),
            Image.AFFINE,
            (1.0 / scale, 0.0, x0,
             0.0, 1.0 / scale, y0),
            resample=Image.LANCZOS,
        )
        frames.append(pil_img)

    return frames


def crossfade_scenes(scene_list: list, crossfade: int = 18) -> list:
    """Blend consecutive scenes with a smooth crossfade transition."""
    if not scene_list:
        return []
    result = list(scene_list[0])
    for i in range(1, len(scene_list)):
        curr = scene_list[i]
        cf = min(crossfade, len(result), len(curr))
        blended = []
        for j in range(cf):
            alpha = j / cf  # 0.0 (prev) → 1.0 (curr)
            prev_arr = np.array(result[-cf + j], dtype=np.float32)
            curr_arr = np.array(curr[j], dtype=np.float32)
            merged = (prev_arr * (1.0 - alpha) + curr_arr * alpha).astype(np.uint8)
            blended.append(Image.fromarray(merged))
        result = result[:-cf] + blended + list(curr[cf:])
    return result


def frames_to_video(frames: list, output_path: str, fps: int = 24):
    w, h = frames[0].size
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "pipe:0",
        "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "18", "-preset", "fast",
        output_path
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for f in frames:
        proc.stdin.write(np.array(f).tobytes())
    proc.stdin.close()
    proc.wait()


def transcribe_words(audio_path: str) -> list:
    model = get_whisper()
    segments, _ = model.transcribe(audio_path, word_timestamps=True, language="en")
    words = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                words.append({"word": w.word.strip(), "start": w.start, "end": w.end})
    return words


def extend_word_gaps(words: list) -> list:
    result = []
    for i, w in enumerate(words):
        end = words[i + 1]["start"] if i + 1 < len(words) else w["end"]
        result.append({"word": w["word"], "start": w["start"], "end": end})
    return result


def words_to_ass(words: list, duration: float, font_name: str = "Arial",
                 font_size: int = 72, output_path: str = None) -> str:
    if output_path is None:
        output_path = tempfile.mktemp(suffix=".ass")

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 720
PlayResY: 1280
Timer: 100.0000

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,2,20,20,512,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    def ts(s: float) -> str:
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        sec = s % 60
        return f"{h}:{m:02d}:{sec:05.2f}"

    lines = []
    chunk_size = 5
    chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]

    for chunk in chunks:
        if not chunk:
            continue
        for wi, active_word in enumerate(chunk):
            parts = []
            for j, w in enumerate(chunk):
                if j == wi:
                    parts.append(f"{{\\c&H00FFFF&}}{w['word']}{{\\c&HFFFFFF&}}")
                else:
                    parts.append(w["word"])
            lines.append(
                f"Dialogue: 0,{ts(active_word['start'])},{ts(active_word['end'])},Default,,0,0,0,,{' '.join(parts)}"
            )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(lines) + "\n")

    return output_path


def mix_audio(voice_path: str, music_file: str, duration: float,
              music_volume: float = 0.15) -> str:
    output = tempfile.mktemp(suffix=".wav")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", voice_path,
        "-stream_loop", "-1", "-i", music_file,
        "-filter_complex",
        f"[1:a]volume={music_volume}[music];[0:a][music]amix=inputs=2:duration=first:normalize=0[out]",
        "-map", "[out]",
        "-t", str(duration),
        "-ar", "22050",
        output
    ], check=True, stderr=subprocess.DEVNULL)
    return output


def merge_video_audio(video_path: str, audio_path: str, ass_path: str,
                      duration: float, output_path: str, font_name: str = "Arial"):
    ass_escaped = ass_path.replace("\\", "/").replace(":", "\\:")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-vf", f"subtitles={ass_escaped}:force_style='FontName={font_name}'",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k",
        "-t", str(duration),
        "-movflags", "+faststart",
        output_path
    ], check=True, stderr=subprocess.PIPE)


# ── Main job function ────────────────────────────────────────────────

def run_job(job_input: dict) -> dict:
    """
    job_input keys:
        narration   (str)  - full narration text
        scenes      (list) - list of image prompts
        title       (str)  - video title
        voice       (str, optional) - Kokoro voice, default "af_heart"
        music_file  (str, optional) - filename in MUSIC_DIR, random if omitted
        font_name   (str, optional) - subtitle font, default "Arial"
        font_size   (int, optional) - subtitle size, default 52
        job_id      (str, optional) - auto-generated if missing
    """
    job_id = job_input.get("job_id") or str(uuid.uuid4())[:8]
    narration = job_input["narration"]
    scenes = job_input["scenes"]
    title = job_input.get("title", "video")
    voice = job_input.get("voice", "af_heart")
    font_name = job_input.get("font_name", "Arial")
    font_size = job_input.get("font_size", 52)

    # Pick music
    music_file_input = job_input.get("music_file")
    if music_file_input:
        music_path = os.path.join(MUSIC_DIR, music_file_input)
    else:
        available = [f for f in os.listdir(MUSIC_DIR) if f.endswith((".mp3", ".wav"))] if os.path.isdir(MUSIC_DIR) else []
        music_path = os.path.join(MUSIC_DIR, random.choice(available)) if available else None

    # Output dimensions
    OUT_W, OUT_H = 720, 1280
    # Pan workspace comes from zoom (no upscale — native SDXL pixels only).
    # zoom_start=1.10 → zoom_end=1.30 gives 47px→83px H spare, 84px→148px V spare.
    # pan_x=0.10 → total 72px source pan, well within limits at all t. ✓
    # Per-frame visible output motion ≈ 0.7px/frame → smooth cinematic drift.
    ZOOM_START = 1.10
    ZOOM_END   = 1.30
    PAN_SEQUENCE = [
        ( 0.10,  0.0),   # pan right + slow zoom in
        (-0.10,  0.0),   # pan left  + slow zoom in
        ( 0.0,  -0.09),  # pan up    + slow zoom in
        ( 0.0,   0.09),  # pan down  + slow zoom in
        ( 0.07, -0.07),  # diagonal  + slow zoom in
    ]
    CROSSFADE = 18  # frames blended at each scene boundary
    fps = 24

    work_dir = tempfile.mkdtemp(prefix=f"job_{job_id}_")
    print(f"[JOB {job_id}] Starting — {len(scenes)} scenes")

    # ── STEP 1: TTS ──────────────────────────────────────────────────
    print(f"[JOB {job_id}] Step 1: TTS ({voice})")
    tts = get_tts()
    audio_path = os.path.join(work_dir, "narration.wav")

    # Sanitize: Kokoro TTS is English-only; strip non-ASCII to prevent silent failure
    tts_text = re.sub(r'[^\x00-\x7F]+', '', narration)
    tts_text = re.sub(r'\s+', ' ', tts_text).strip()
    if not tts_text:
        raise RuntimeError(f"Narration is empty after ASCII sanitization. Original: {narration[:80]!r}")
    print(f"[JOB {job_id}] TTS text ({len(tts_text)} chars): {tts_text[:80]}...")

    samples_list = []
    for _, _, audio in tts(tts_text, voice=voice, speed=1.0):
        if audio is not None:
            samples_list.append(audio)
    if not samples_list:
        raise RuntimeError(f"TTS produced no audio for text: {tts_text[:80]!r}")
    audio_tensor = torch.cat([
        s.detach().cpu() if isinstance(s, torch.Tensor) else torch.tensor(s)
        for s in samples_list
    ])
    torchaudio.save(audio_path, audio_tensor.unsqueeze(0), 24000)
    duration = sf.info(audio_path).duration
    print(f"[JOB {job_id}] Audio duration: {duration:.2f}s")

    # ── STEP 2: Transcribe for subtitles ─────────────────────────────
    print(f"[JOB {job_id}] Step 2: Whisper transcription")
    words = extend_word_gaps(transcribe_words(audio_path))

    # ── STEP 3: Generate images ───────────────────────────────────────
    print(f"[JOB {job_id}] Step 3: Image generation ({len(scenes)} scenes)")
    pipe = get_sdxl()
    num_scenes = len(scenes)
    # Extra frames per scene to compensate for crossfade overlap
    scene_frames = int((duration / num_scenes) * fps) + CROSSFADE

    scene_frames_list = []
    for idx, prompt in enumerate(scenes):
        print(f"[JOB {job_id}]   Scene {idx+1}/{num_scenes}")
        img_array = generate_image(prompt, pipe)  # 720x1280 native SDXL pixels
        # No upscale — zoom provides pan workspace, avoids double-LANCZOS artifacts
        pan = PAN_SEQUENCE[idx % len(PAN_SEQUENCE)]
        frames = ken_burns(img_array, scene_frames,
                           out_w=OUT_W, out_h=OUT_H,
                           zoom_start=ZOOM_START, zoom_end=ZOOM_END,
                           pan_x=pan[0], pan_y=pan[1])
        scene_frames_list.append(frames)

    # Crossfade between scenes for smooth transitions
    all_frames = crossfade_scenes(scene_frames_list, CROSSFADE)
    # Ensure enough frames to fill full audio duration
    needed = int(duration * fps) + fps
    while len(all_frames) < needed:
        all_frames.append(all_frames[-1])

    # ── STEP 4: Silent video ──────────────────────────────────────────
    print(f"[JOB {job_id}] Step 4: Writing video")
    silent_video = os.path.join(work_dir, "silent.mp4")
    frames_to_video(all_frames, silent_video, fps=fps)

    # ── STEP 5: Subtitles ─────────────────────────────────────────────
    print(f"[JOB {job_id}] Step 5: Subtitles")
    ass_path = os.path.join(work_dir, "subs.ass")
    words_to_ass(words, duration, font_name=font_name, font_size=font_size,
                 output_path=ass_path)

    # ── STEP 6: Mix audio ─────────────────────────────────────────────
    print(f"[JOB {job_id}] Step 6: Audio mix")
    if music_path and os.path.exists(music_path):
        final_audio = mix_audio(audio_path, music_path, duration)
    else:
        final_audio = audio_path

    # ── STEP 7: Final merge ───────────────────────────────────────────
    print(f"[JOB {job_id}] Step 7: Final merge")
    safe_title = "".join(c for c in title if c.isalnum() or c in " _-")[:50].strip()
    output_path = os.path.join(OUTPUT_DIR, f"{job_id}_{safe_title}.mp4")
    merge_video_audio(silent_video, final_audio, ass_path, duration,
                      output_path, font_name=font_name)

    # ── STEP 8: Upload video ──────────────────────────────────────────
    print(f"[JOB {job_id}] Step 8: Uploading video")
    video_url = None
    upload_errors = []

    # Try catbox.moe (permanent, no account needed, 200MB limit)
    r1 = subprocess.run(
        ["curl", "-s", "--max-time", "120",
         "-F", "reqtype=fileupload",
         "-F", f"fileToUpload=@{output_path}",
         "https://catbox.moe/user/api.php"],
        capture_output=True, text=True
    )
    if r1.stdout.strip().startswith("http"):
        video_url = r1.stdout.strip()
        print(f"[JOB {job_id}] Uploaded to catbox.moe")
    else:
        upload_errors.append(f"catbox: rc={r1.returncode} out={r1.stdout[:100]}")

    # Fallback 1: litterbox.catbox.moe (temporary 72h)
    if not video_url:
        r2 = subprocess.run(
            ["curl", "-s", "--max-time", "120",
             "-F", "reqtype=fileupload",
             "-F", "time=72h",
             "-F", f"fileToUpload=@{output_path}",
             "https://litterbox.catbox.moe/resources/internals/api.php"],
            capture_output=True, text=True
        )
        if r2.stdout.strip().startswith("http"):
            video_url = r2.stdout.strip()
            print(f"[JOB {job_id}] Uploaded to litterbox")
        else:
            upload_errors.append(f"litterbox: rc={r2.returncode} out={r2.stdout[:100]}")

    # Fallback 2: 0x0.st (512MB limit, long-term hosting)
    if not video_url:
        r3 = subprocess.run(
            ["curl", "-s", "--max-time", "120",
             "-F", f"file=@{output_path}",
             "https://0x0.st"],
            capture_output=True, text=True
        )
        if r3.stdout.strip().startswith("http"):
            video_url = r3.stdout.strip()
            print(f"[JOB {job_id}] Uploaded to 0x0.st")
        else:
            upload_errors.append(f"0x0: rc={r3.returncode} out={r3.stdout[:100]}")

    # Fallback 3: gofile.io (no size limit, free)
    if not video_url:
        try:
            import urllib.request
            srv_resp = urllib.request.urlopen("https://api.gofile.io/servers", timeout=10)
            import json as _json
            srv_data = _json.loads(srv_resp.read().decode())
            server = srv_data["data"]["servers"][0]["name"]
            r4 = subprocess.run(
                ["curl", "-s", "--max-time", "180",
                 "-F", f"file=@{output_path}",
                 f"https://{server}.gofile.io/uploadFile"],
                capture_output=True, text=True
            )
            r4_json = _json.loads(r4.stdout)
            if r4_json.get("status") == "ok":
                video_url = r4_json["data"]["downloadPage"]
                print(f"[JOB {job_id}] Uploaded to gofile.io")
            else:
                upload_errors.append(f"gofile: {r4.stdout[:100]}")
        except Exception as e:
            upload_errors.append(f"gofile: {str(e)[:100]}")

    if not video_url:
        raise RuntimeError(f"All uploads failed: {'; '.join(upload_errors)}")
    print(f"[JOB {job_id}] Done → {video_url}")

    return {
        "job_id": job_id,
        "video_url": video_url,
        "duration": round(duration, 2),
        "scenes": num_scenes,
    }
