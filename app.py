import os
import uuid
import subprocess
import tempfile
import numpy as np
import soundfile as sf
import random
import urllib.request
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
              zoom_start=1.0, zoom_end=1.08, pan_x=0.0, pan_y=0.0) -> list:
    h, w = frame_array.shape[:2]
    frames = []
    for i in range(total_frames):
        t = i / max(total_frames - 1, 1)
        scale = zoom_start + (zoom_end - zoom_start) * t
        new_w = int(w / scale)
        new_h = int(h / scale)
        cx = int(w / 2 + pan_x * w * t)
        cy = int(h / 2 + pan_y * h * t)
        x1 = max(0, cx - new_w // 2)
        y1 = max(0, cy - new_h // 2)
        x2 = min(w, x1 + new_w)
        y2 = min(h, y1 + new_h)
        if x2 - x1 < new_w:
            x1 = max(0, x2 - new_w)
        if y2 - y1 < new_h:
            y1 = max(0, y2 - new_h)
        cropped = frame_array[y1:y2, x1:x2]
        pil_img = Image.fromarray(cropped).resize((w, h), Image.LANCZOS)
        frames.append(pil_img)
    return frames


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
Style: Default,{font_name},{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,2,20,20,80,1

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
              music_volume: float = 0.12) -> str:
    output = tempfile.mktemp(suffix=".wav")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", voice_path,
        "-stream_loop", "-1", "-i", music_file,
        "-filter_complex",
        f"[1:a]volume={music_volume}[music];[0:a][music]amix=inputs=2:duration=first[out]",
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
        font_size   (int, optional) - subtitle size, default 72
        job_id      (str, optional) - auto-generated if missing
    """
    job_id = job_input.get("job_id") or str(uuid.uuid4())[:8]
    narration = job_input["narration"]
    scenes = job_input["scenes"]
    title = job_input.get("title", "video")
    voice = job_input.get("voice", "af_heart")
    font_name = job_input.get("font_name", "Arial")
    font_size = job_input.get("font_size", 72)
    fps = 24

    # Pick music
    music_file_input = job_input.get("music_file")
    if music_file_input:
        music_path = os.path.join(MUSIC_DIR, music_file_input)
    else:
        available = [f for f in os.listdir(MUSIC_DIR) if f.endswith((".mp3", ".wav"))] if os.path.isdir(MUSIC_DIR) else []
        music_path = os.path.join(MUSIC_DIR, random.choice(available)) if available else None

    work_dir = tempfile.mkdtemp(prefix=f"job_{job_id}_")
    print(f"[JOB {job_id}] Starting — {len(scenes)} scenes")

    # ── STEP 1: TTS ──────────────────────────────────────────────────
    print(f"[JOB {job_id}] Step 1: TTS ({voice})")
    tts = get_tts()
    audio_path = os.path.join(work_dir, "narration.wav")
    samples_list = []
    for _, _, audio in tts(narration, voice=voice, speed=1.0):
        if audio is not None:
            samples_list.append(audio)
    if not samples_list:
        raise RuntimeError("TTS produced no audio")
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
    scene_frames = int((duration / num_scenes) * fps)
    pan_options = [(0.02, 0.0), (-0.02, 0.0), (0.0, 0.02), (0.0, -0.02)]

    all_frames = []
    for idx, prompt in enumerate(scenes):
        print(f"[JOB {job_id}]   Scene {idx+1}/{num_scenes}")
        img_array = generate_image(prompt, pipe)
        pan = random.choice(pan_options)
        frames = ken_burns(img_array, scene_frames, zoom_start=1.0, zoom_end=1.08,
                           pan_x=pan[0], pan_y=pan[1])
        all_frames.extend(frames)

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

    # ── STEP 8: Upload to transfer.sh ────────────────────────────────
    print(f"[JOB {job_id}] Step 8: Uploading to transfer.sh")
    filename = os.path.basename(output_path).replace(" ", "_")
    upload_url = f"https://transfer.sh/{filename}"
    req = urllib.request.Request(
        upload_url,
        data=open(output_path, "rb"),
        method="PUT",
        headers={"Max-Days": "3"},
    )
    with urllib.request.urlopen(req) as resp:
        video_url = resp.read().decode("utf-8").strip()
    print(f"[JOB {job_id}] Done → {video_url}")

    return {
        "job_id": job_id,
        "video_url": video_url,
        "duration": round(duration, 2),
        "scenes": num_scenes,
    }
