"""ComfyUI process renderer + OpenAI-shaped /v1 probe wrapper.

MiniMax H3 on a 24 GB Ampere card uses the official Comfy-Org pruned INT8
FL2VA pack (ComfyUI 0.30+ native nodes, PyTorch CUDA 13 for INT8 convrot).
ComfyUI itself does not serve ``/v1/models``; this wrapper starts ComfyUI
and answers Cortex-deployer health on the recipe port.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .base import ProcessLaunch, _model_path
from ..spec import Recipe

DIT = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
TURBO_LORA = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"

# Open H3-Base is a 768p-class canvas (official 16:9 native 1344×768).
# 864×480 is the fast 24 GB preview. 1920×1088 is 1080p-class stretch on
# the same DiT — slower, more VRAM. 2K regenerate / 4K are NOT in the
# local weights (H3-Regenerate-2K is still API-only).
# Duration: official H3-Base trains ~4–15 s (124–362 frames at 24 fps).
# 15 s 480p is proven on the 3090 (~19.8 GB). 60 s is ~4× that frame
# count — out of distribution, not a cost problem.
MIN_SECONDS = 4
MAX_SECONDS = 15
MAX_COMPOSE_SECONDS = 3600
SEGMENT_TIMEOUT_S = 45 * 60
MAX_EDGE = 1920
SIZE_PRESETS: dict[str, tuple[int, int]] = {
    "16:9": (864, 480),
    "9:16": (480, 864),
    "1:1": (640, 640),
    "4:3": (768, 576),
    "480p": (864, 480),
    "480p:16:9": (864, 480),
    "480p:9:16": (480, 864),
    "480p:1:1": (640, 640),
    "480p:4:3": (768, 576),
    "720p": (1344, 768),
    "720p:16:9": (1344, 768),
    "720p:9:16": (768, 1344),
    "720p:1:1": (768, 768),
    "720p:4:3": (1024, 768),
    "1080p": (1920, 1088),
    "1080p:16:9": (1920, 1088),
    "1080p:9:16": (1088, 1920),
    "1080p:1:1": (1088, 1088),
    "1080p:4:3": (1440, 1088),
}
H3_CAPABILITIES = {
    "seconds": [4, 5, 6, 8, 10, 12, 15, 30, 60, 120, 300, 900, 1800, 3600],
    "resolutions": [
        {"id": "480p", "label": "480p", "hint": "Fast preview"},
        {"id": "720p", "label": "720p", "hint": "Native H3-Base 1344×768"},
        {"id": "1080p", "label": "1080p", "hint": "Same DiT, slower on 24 GB"},
    ],
    "aspects": [
        {"id": "16:9", "label": "Landscape"},
        {"id": "9:16", "label": "Portrait"},
        {"id": "1:1", "label": "Square"},
        {"id": "4:3", "label": "Classic"},
    ],
    "sizes": {
        "480p": {"16:9": "864x480", "9:16": "480x864", "1:1": "640x640", "4:3": "768x576"},
        "720p": {"16:9": "1344x768", "9:16": "768x1344", "1:1": "768x768", "4:3": "1024x768"},
        "1080p": {"16:9": "1920x1088", "9:16": "1088x1920", "1:1": "1088x1088", "4:3": "1440x1088"},
    },
    "max_edge": MAX_EDGE,
    "first_frame": True,
    "last_frame": True,
    "input_video": True,
    "turbo": True,
    "seed": True,
    "note": "Each shot is 4–15s. Longer clips are stitched from connected shots, up to 1 hour.",
}


def render_comfyui(recipe: Recipe) -> ProcessLaunch:
    if recipe.engine != "comfyui":
        raise ValueError("comfyui renderer received a different engine")
    root = str(Path(_model_path(recipe)).expanduser())
    argv = [
        sys.executable,
        "-m",
        "cortex_deployer.engines.comfyui",
        "--comfy-root",
        root,
        "--host",
        recipe.launch.host,
        "--port",
        str(recipe.launch.port),
        "--served-name",
        recipe.model.served_name,
    ]
    argv.extend(recipe.launch.extra_args)
    return ProcessLaunch(
        argv=tuple(argv),
        env=recipe.launch.env,
        host=recipe.launch.host,
        port=recipe.launch.port,
        engine="comfyui",
    )


def h3_frame_length(duration_s: float) -> int:
    """Official H3 17k+5 grid at 24 fps (ComfyUI T2V Math Expression)."""
    frames = max(5, int(round(float(duration_s) * 24)))
    return frames + (5 - (frames % 17)) % 17


def _align32(n: int, lo: int = 32, hi: int = MAX_EDGE) -> int:
    n = max(lo, min(hi, int(n)))
    n -= n % 32
    return max(lo, n)


def parse_size(
    size: str,
    default: tuple[int, int] = (864, 480),
    *,
    resolution: str = "",
    aspect: str = "",
) -> tuple[int, int]:
    raw = (size or "").lower().replace(" ", "")
    res = (resolution or "").lower().strip()
    asp = (aspect or "").lower().strip()
    for key in (f"{res}:{asp}", res, f"{raw}:{asp}", raw, asp):
        if key and key in SIZE_PRESETS:
            return SIZE_PRESETS[key]
    if "x" not in raw:
        return default
    left, right = raw.split("x", 1)
    try:
        width, height = int(left), int(right)
    except ValueError:
        return default
    if width < 32 or height < 32:
        return default
    return _align32(width), _align32(height)


def max_seconds_for_size(width: int, height: int) -> float:
    """Longer clips at 720p/1080p blow the 24 GB activation envelope.

    15 s at 480p (362 frames, ~19.8 GB peak, 20-step) is the documented
    3090 ceiling. 720p is ~2.5× the spatial tokens, so stay at 10 s.
    """
    megapixels = (int(width) * int(height)) / 1_000_000
    if megapixels >= 1.8:
        return 6.0
    if megapixels >= 0.8:
        return 10.0
    return float(MAX_SECONDS)


def parse_seconds(value, default: float = 5.0, *, width: int = 864, height: int = 480) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        seconds = default
    ceiling = min(float(MAX_SECONDS), max_seconds_for_size(width, height))
    if seconds < MIN_SECONDS:
        return float(MIN_SECONDS)
    if seconds > ceiling:
        return ceiling
    return seconds


def parse_target_seconds(value, default: float = 5.0) -> float:
    """Requested clip length, including stitched jobs up to 1 hour."""
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        seconds = default
    if seconds < MIN_SECONDS:
        return float(MIN_SECONDS)
    if seconds > MAX_COMPOSE_SECONDS:
        return float(MAX_COMPOSE_SECONDS)
    return seconds


def archive_dir() -> Path:
    path = Path(os.environ.get("CORTEX_VIDEO_ARCHIVE") or "~/.cortex-deployer/video-archive").expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def archive_mp4_path(vid: str) -> Path:
    return archive_dir() / f"{vid}.mp4"


def archive_meta_path(vid: str) -> Path:
    return archive_dir() / f"{vid}.json"


def write_archive(meta: dict[str, Any], blob: bytes | None = None) -> dict[str, Any]:
    vid = str(meta.get("id") or "")
    if not vid:
        return meta
    stored = dict(meta)
    stored["archived"] = True
    if blob:
        archive_mp4_path(vid).write_bytes(blob)
        stored["status"] = "completed"
        stored["url"] = f"/v1/videos/{vid}/content"
    archive_meta_path(vid).write_text(json.dumps(stored), encoding="utf-8")
    return stored


def load_archives() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in archive_dir().glob("*.json"):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(meta, dict) and meta.get("id"):
            vid = str(meta["id"])
            if archive_mp4_path(vid).is_file():
                meta["status"] = "completed"
                meta["archived"] = True
                meta["url"] = f"/v1/videos/{vid}/content"
            out[vid] = meta
    return out


def import_comfy_mp4s(comfy_root: Path) -> dict[str, dict[str, Any]]:
    """Keep already-rendered Comfy outputs playable after history is wiped."""
    imported: dict[str, dict[str, Any]] = {}
    output = Path(comfy_root) / "output"
    if not output.is_dir():
        return imported
    existing = {p.name for p in archive_dir().glob("*.mp4")}
    for mp4 in sorted(output.rglob("*.mp4")):
        vid = "file-" + "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in mp4.stem).strip("-_")
        if not vid or f"{vid}.mp4" in existing:
            continue
        try:
            shutil.copy2(mp4, archive_mp4_path(vid))
        except OSError:
            continue
        meta = {
            "id": vid,
            "object": "video",
            "status": "completed",
            "prompt": mp4.stem.replace("_", " ").strip() or "Clip",
            "submitted_at": mp4.stat().st_mtime,
            "archived": True,
            "url": f"/v1/videos/{vid}/content",
        }
        archive_meta_path(vid).write_text(json.dumps(meta), encoding="utf-8")
        imported[vid] = meta
        existing.add(f"{vid}.mp4")
    return imported


def fetch_comfy_asset(comfy_base: str, asset: dict[str, str]) -> bytes | None:
    filename = str(asset.get("filename") or "")
    if not filename:
        return None
    qs = (
        f"filename={filename}"
        f"&subfolder={asset.get('subfolder') or ''}"
        f"&type={asset.get('type') or 'output'}"
    )
    try:
        req = Request(f"{comfy_base}/view?{qs}")
        with urlopen(req, timeout=60) as resp:
            return resp.read()
    except (URLError, TimeoutError, OSError, HTTPError):
        return None


def extract_last_frame(mp4: Path, jpeg: Path) -> None:
    jpeg.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-sseof", "-0.05", "-i", str(mp4), "-frames:v", "1", str(jpeg)],
        check=True,
        capture_output=True,
        timeout=60,
    )


def concat_mp4s(parts: list[Path], dest: Path) -> None:
    listing = dest.with_suffix(".txt")
    listing.write_text("".join(f"file '{p.resolve()}'\n" for p in parts), encoding="utf-8")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(dest)],
            check=True,
            capture_output=True,
            timeout=300,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dest)],
            check=True,
            capture_output=True,
            timeout=600,
        )
    finally:
        listing.unlink(missing_ok=True)


def decode_image_payload(value) -> tuple[bytes, str] | None:
    """Accept a data URL, raw base64, or {b64_json, mime} dict."""
    mime = "image/jpeg"
    raw = ""
    if isinstance(value, dict):
        raw = str(value.get("b64_json") or value.get("data") or "")
        mime = str(value.get("mime") or value.get("content_type") or mime)
    elif isinstance(value, str):
        raw = value.strip()
    else:
        return None
    if not raw:
        return None
    if raw.startswith("data:"):
        header, _, payload = raw.partition(",")
        if not payload:
            return None
        if "image/" in header:
            mime = header[5:].split(";")[0] or mime
        raw = payload
    try:
        blob = base64.b64decode(raw, validate=False)
    except (ValueError, TypeError):
        return None
    if len(blob) < 32 or len(blob) > 8 * 1024 * 1024:
        return None
    return blob, mime


def t2v_api_prompt(
    prompt: str,
    *,
    width: int = 864,
    height: int = 480,
    duration_s: float = 5.0,
    seed: int = 1,
    turbo: bool = True,
    steps: int | None = None,
    first_image_name: str | None = None,
    last_image_name: str | None = None,
) -> dict[str, Any]:
    """API-format graph matching the official MiniMax H3 T2V template nodes."""
    length = h3_frame_length(duration_s)
    use_steps = int(steps if steps is not None else (8 if turbo else 20))
    model_src = ["lora", 0] if turbo else ["unet", 0]
    graph: dict[str, Any] = {
        "unet": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": DIT, "weight_dtype": "default"},
        },
        "clip": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": TEXT_ENCODER,
                "type": "minimax",
                "device": "default",
            },
        },
        "vvae": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "avae": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
        "i2v": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "clip": ["clip", 0],
                "vae": ["vvae", 0],
                "prompt": prompt,
                "width": int(width),
                "height": int(height),
                "length": length,
            },
        },
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}},
        "sampler": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "res_multistep"},
        },
        "sched": {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": model_src,
                "scheduler": "simple",
                "steps": use_steps,
                "denoise": 1.0,
            },
        },
        "guider": {
            "class_type": "BasicGuider",
            "inputs": {"model": model_src, "conditioning": ["i2v", 0]},
        },
        "sample": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["noise", 0],
                "guider": ["guider", 0],
                "sampler": ["sampler", 0],
                "sigmas": ["sched", 0],
                "latent_image": ["i2v", 1],
            },
        },
        "decode": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["sample", 0], "vae": ["vvae", 0]},
        },
        "adecode": {
            "class_type": "VAEDecodeAudio",
            "inputs": {"samples": ["sample", 0], "vae": ["avae", 0]},
        },
        "video": {
            "class_type": "CreateVideo",
            "inputs": {
                "images": ["decode", 0],
                "audio": ["adecode", 0],
                "fps": 24,
                "bit_depth": 8,
            },
        },
        "save": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["video", 0],
                "filename_prefix": "video/MiniMax_H3",
                "format": "auto",
                "codec": "auto",
            },
        },
    }
    if turbo:
        graph["lora"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["unet", 0],
                "lora_name": TURBO_LORA,
                "strength_model": 1.0,
            },
        }
    if first_image_name:
        graph["first_img"] = {
            "class_type": "LoadImage",
            "inputs": {"image": first_image_name},
        }
        graph["i2v"]["inputs"]["first_frame"] = ["first_img", 0]
    if last_image_name:
        graph["last_img"] = {
            "class_type": "LoadImage",
            "inputs": {"image": last_image_name},
        }
        graph["i2v"]["inputs"]["last_frame"] = ["last_img", 0]
    return graph


def comfy_python(comfy_root: Path) -> str:
    env = os.environ.get("CORTEX_DEPLOYER_COMFYUI_PYTHON")
    if env and Path(env).is_file():
        return env
    for cand in (
        comfy_root / ".venv" / "bin" / "python",
        comfy_root / "venv" / "bin" / "python",
        comfy_root / ".venv" / "Scripts" / "python.exe",
        comfy_root / "venv" / "Scripts" / "python.exe",
    ):
        if cand.is_file():
            return str(cand)
    raise FileNotFoundError(
        f"ComfyUI venv python not found under {comfy_root}. "
        "Create .venv with torch CUDA 13 or set CORTEX_DEPLOYER_COMFYUI_PYTHON"
    )


def _comfy_base(host: str, port: int) -> str:
    h = "127.0.0.1" if host in {"0.0.0.0", "::", "[::]"} else host
    return f"http://{h}:{port}"


def _wait_comfy(url: str, timeout: float = 300.0) -> None:
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        try:
            req = Request(url, headers={"accept": "application/json"}, method="GET")
            with urlopen(req, timeout=2.0) as resp:
                if 200 <= getattr(resp, "status", 200) < 300:
                    return
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            last = str(exc)
        time.sleep(0.5)
    raise TimeoutError(f"ComfyUI did not become ready at {url}: {last}")


def _spawn_comfy(comfy_root: Path, host: str, port: int, extra: list[str]) -> subprocess.Popen[Any]:
    main = comfy_root / "main.py"
    if not main.is_file():
        raise FileNotFoundError(f"ComfyUI main.py not found under {comfy_root}")
    python = comfy_python(comfy_root)
    argv = [
        python,
        str(main),
        "--listen",
        host,
        "--port",
        str(port),
        "--disable-auto-launch",
        *extra,
    ]
    return subprocess.Popen(
        argv,
        cwd=str(comfy_root),
        stdin=subprocess.DEVNULL,
        stdout=sys.stdout,
        stderr=sys.stderr,
        start_new_session=True,
    )


def _stop_comfy(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except OSError:
        proc.terminate()
    try:
        proc.wait(timeout=15)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()


def _ext_for_mime(mime: str) -> str:
    mime = (mime or "").lower()
    if "png" in mime:
        return "png"
    if "webp" in mime:
        return "webp"
    if "gif" in mime:
        return "gif"
    return "jpg"


def _comfy_upload_image(comfy_base: str, blob: bytes, mime: str) -> str:
    filename = f"h3-{uuid.uuid4().hex[:12]}.{_ext_for_mime(mime)}"
    boundary = f"----CortexH3{uuid.uuid4().hex}"
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
        f"Content-Type: {mime or 'image/jpeg'}\r\n\r\n"
    ).encode()
    footer = (
        f"\r\n--{boundary}\r\n"
        'Content-Disposition: form-data; name="overwrite"\r\n\r\n'
        f"true\r\n--{boundary}--\r\n"
    ).encode()
    body = header + blob + footer
    req = Request(
        f"{comfy_base}/upload/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode() or "{}")
    name = str(payload.get("name") or filename)
    sub = str(payload.get("subfolder") or "")
    return f"{sub}/{name}" if sub else name


def _frame_from_body(body: dict, *keys: str) -> tuple[bytes, str] | None:
    for key in keys:
        decoded = decode_image_payload(body.get(key))
        if decoded:
            return decoded
    return None


def _http_json(url: str, *, data: bytes | None = None, timeout: float = 30.0) -> tuple[int, bytes, str]:
    headers = {"accept": "application/json"}
    method = "GET"
    if data is not None:
        headers["content-type"] = "application/json"
        method = "POST"
    req = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=timeout) as resp:
            ctype = resp.headers.get("content-type") or "application/json"
            return getattr(resp, "status", 200), resp.read(), ctype
    except HTTPError as exc:
        body = exc.read() if exc.fp else str(exc).encode()
        return int(exc.code), body, "application/json"


SUBMIT_GRACE_S = 20.0


def comfy_queue_prompt_ids(queue: dict[str, Any] | None) -> tuple[set[str], set[str]]:
    """Split Comfy ``/queue`` into running vs pending prompt ids."""
    running: set[str] = set()
    pending: set[str] = set()

    def _take(bucket: Any, dest: set[str]) -> None:
        if not isinstance(bucket, list):
            return
        for item in bucket:
            pid = ""
            if isinstance(item, (list, tuple)) and len(item) > 1:
                pid = str(item[1] or "")
            elif isinstance(item, dict):
                pid = str(item.get("prompt_id") or item.get("id") or "")
            if pid:
                dest.add(pid)

    if isinstance(queue, dict):
        _take(queue.get("queue_running"), running)
        _take(queue.get("queue_pending"), pending)
    return running, pending


def resolve_video_status(
    prompt_id: str,
    *,
    local: dict[str, Any] | None = None,
    history: dict[str, Any] | None = None,
    queue: dict[str, Any] | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Map Comfy history/queue onto an OpenAI-shaped video status.

    Comfy ``GET /history/{id}`` returns ``200 {}`` for unknown *and* still-
    running ids. Treating that as ``queued`` made Studio spin forever: a
    live graph never left queued, and a wrapper restart turned finished
    clips into queued ghosts.
    """
    pid = str(prompt_id or "")
    meta = dict(local or {"id": pid})
    meta["id"] = pid
    clock = time.time() if now is None else float(now)
    if meta.get("archived") and archive_mp4_path(pid).is_file():
        meta["status"] = "completed"
        meta["url"] = f"/v1/videos/{pid}/content"
        return meta

    # Stitched jobs use a parent UUID Comfy never sees. The composer thread
    # owns that row; mapping the parent through /history/{id} + /queue is
    # what marked 30s Studio clips failed after SUBMIT_GRACE_S while the
    # GPU was still rendering a different prompt_id.
    if local and local.get("compose"):
        status = str(local.get("status") or "queued").lower() or "queued"
        meta["status"] = status
        if status not in {"failed", "error", "cancelled", "canceled"}:
            meta.pop("error", None)
        return meta

    hist = None
    if isinstance(history, dict):
        candidate = history.get(pid)
        if isinstance(candidate, dict):
            hist = candidate
        elif "outputs" in history or "status" in history:
            hist = history

    if isinstance(hist, dict):
        status = hist.get("status") or {}
        if isinstance(status, dict) and status.get("completed"):
            meta["status"] = "completed"
            asset = _history_asset(hist)
            if asset:
                meta["asset"] = asset
                meta["url"] = f"/v1/videos/{pid}/content"
            return meta
        if isinstance(status, dict) and str(status.get("status_str") or "").lower() == "error":
            meta["status"] = "failed"
            meta["error"] = status.get("messages") or "comfyui error"
            return meta
        meta["status"] = "in_progress"
        return meta

    running, pending = comfy_queue_prompt_ids(queue)
    if pid in running:
        meta["status"] = "in_progress"
        return meta
    if pid in pending:
        meta["status"] = "queued"
        return meta

    submitted = float(meta.get("submitted_at") or 0)
    if local and submitted and (clock - submitted) < SUBMIT_GRACE_S:
        meta["status"] = "queued"
        return meta

    meta["status"] = "failed"
    meta["error"] = "clip is not on the engine (dropped or never started)"
    return meta


def apply_resolved_status(current: dict[str, Any] | None, resolved: dict[str, Any]) -> dict[str, Any]:
    """Write a poll mapping onto the in-memory row.

    GET /v1/videos/{parent} must not stamp ``failed`` onto a live stitch:
    that is how Studio showed "no longer available" while Comfy kept
    generating segment N under a different prompt id.
    """
    cur = dict(current or {})
    meta = dict(resolved or {})
    live = str(cur.get("status") or "") in {"queued", "in_progress"}
    if cur.get("compose") and live and str(meta.get("status") or "") == "failed":
        return cur
    return {**cur, **meta}


def _history_asset(history: dict[str, Any]) -> dict[str, str] | None:
    for node_out in (history.get("outputs") or {}).values():
        if not isinstance(node_out, dict):
            continue
        for key in ("videos", "gifs", "images", "files"):
            items = node_out.get(key)
            if isinstance(items, list) and items:
                first = items[0]
                if isinstance(first, dict) and first.get("filename"):
                    return {
                        "filename": str(first.get("filename") or ""),
                        "subfolder": str(first.get("subfolder") or ""),
                        "type": str(first.get("type") or "output"),
                    }
    return None


def wait_comfy_prompt(comfy_base: str, prompt_id: str, *, timeout: float = SEGMENT_TIMEOUT_S) -> dict[str, Any]:
    local = {"id": prompt_id, "submitted_at": time.time()}
    deadline = time.time() + float(timeout)
    while time.time() < deadline:
        hist_payload: dict[str, Any] = {}
        queue_payload: dict[str, Any] = {}
        code, raw, _ = _http_json(f"{comfy_base}/history/{prompt_id}", timeout=10)
        if code < 400:
            try:
                parsed = json.loads(raw.decode() or "{}")
            except json.JSONDecodeError:
                parsed = {}
            if isinstance(parsed, dict):
                hist_payload = parsed
        qcode, qraw, _ = _http_json(f"{comfy_base}/queue", timeout=5)
        if qcode < 400:
            try:
                qparsed = json.loads(qraw.decode() or "{}")
            except json.JSONDecodeError:
                qparsed = {}
            if isinstance(qparsed, dict):
                queue_payload = qparsed
        meta = resolve_video_status(
            prompt_id, local=local, history=hist_payload, queue=queue_payload,
        )
        if meta.get("status") == "completed":
            return meta
        if meta.get("status") == "failed" and (time.time() - local["submitted_at"]) > SUBMIT_GRACE_S:
            raise RuntimeError(str(meta.get("error") or "segment failed"))
        time.sleep(2)
    raise TimeoutError(f"segment {prompt_id} timed out")


def submit_shot(
    comfy_base: str,
    *,
    prompt: str,
    width: int,
    height: int,
    duration_s: float,
    seed: int,
    turbo: bool,
    first_image_name: str | None,
    last_image_name: str | None,
) -> str:
    graph = t2v_api_prompt(
        prompt,
        width=width,
        height=height,
        duration_s=duration_s,
        seed=seed,
        turbo=turbo,
        first_image_name=first_image_name,
        last_image_name=last_image_name,
    )
    code, payload, _ = _http_json(
        f"{comfy_base}/prompt",
        data=json.dumps({"prompt": graph}).encode(),
        timeout=30,
    )
    try:
        parsed = json.loads(payload.decode() or "{}")
    except json.JSONDecodeError:
        parsed = {}
    prompt_id = str(parsed.get("prompt_id") or "")
    if code >= 400 or not prompt_id:
        raise RuntimeError(str(parsed.get("error") or payload[:200]))
    return prompt_id


def run_compose_job(
    *,
    job_id: str,
    comfy_base: str,
    jobs: dict[str, dict[str, Any]],
    lock: threading.Lock,
    prompt: str,
    width: int,
    height: int,
    target_s: float,
    shot_s: float,
    seed: int,
    turbo: bool,
    first_name: str | None,
) -> None:
    work = archive_dir() / f"compose-{job_id}"
    work.mkdir(parents=True, exist_ok=True)
    n = max(1, int(math.ceil(target_s / shot_s)))
    parts: list[Path] = []
    last_name = first_name
    try:
        for i in range(n):
            remaining = target_s - (shot_s * i)
            dur = min(shot_s, max(float(MIN_SECONDS), remaining))
            with lock:
                row = dict(jobs.get(job_id) or {})
                row.update({"status": "in_progress", "segment": i + 1, "segments": n})
                jobs[job_id] = row
            pid = submit_shot(
                comfy_base,
                prompt=prompt,
                width=width,
                height=height,
                duration_s=dur,
                seed=int(seed) + i,
                turbo=turbo,
                first_image_name=last_name,
                last_image_name=None,
            )
            done = wait_comfy_prompt(comfy_base, pid)
            blob = fetch_comfy_asset(comfy_base, done.get("asset") or {})
            if not blob:
                raise RuntimeError("segment produced no video")
            part = work / f"seg-{i:03d}.mp4"
            part.write_bytes(blob)
            parts.append(part)
            if i < n - 1:
                frame = work / f"last-{i:03d}.jpg"
                extract_last_frame(part, frame)
                last_name = _comfy_upload_image(comfy_base, frame.read_bytes(), "image/jpeg")
        dest = archive_mp4_path(job_id)
        concat_mp4s(parts, dest)
        with lock:
            meta = dict(jobs.get(job_id) or {"id": job_id})
            meta.update({
                "status": "completed",
                "archived": True,
                "url": f"/v1/videos/{job_id}/content",
                "segments": n,
            })
            jobs[job_id] = write_archive(meta, dest.read_bytes())
    except Exception as exc:
        with lock:
            meta = dict(jobs.get(job_id) or {"id": job_id})
            meta["status"] = "failed"
            meta["error"] = str(exc)
            jobs[job_id] = meta
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _handler(served_name: str, comfy_base: str, comfy_proc: subprocess.Popen[Any], comfy_root: Path):
    jobs: dict[str, dict[str, Any]] = load_archives()
    jobs.update(import_comfy_mp4s(comfy_root))
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:  # noqa: A003
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def _send(self, code: int, body: dict | bytes, ctype: str = "application/json") -> None:
            raw = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _video_status(self, prompt_id: str) -> dict[str, Any]:
            with lock:
                local = dict(jobs.get(prompt_id) or {"id": prompt_id, "model": served_name})
            if local.get("compose") and str(local.get("status") or "") in {
                "queued", "in_progress", "failed", "cancelled", "canceled",
            }:
                return local
            hist_payload: dict[str, Any] | None = None
            queue_payload: dict[str, Any] | None = None
            code, raw, _ = _http_json(f"{comfy_base}/history/{prompt_id}", timeout=10)
            if code < 400:
                try:
                    parsed = json.loads(raw.decode() or "{}")
                except json.JSONDecodeError:
                    parsed = {}
                if isinstance(parsed, dict):
                    hist_payload = parsed
            qcode, qraw, _ = _http_json(f"{comfy_base}/queue", timeout=5)
            if qcode < 400:
                try:
                    qparsed = json.loads(qraw.decode() or "{}")
                except json.JSONDecodeError:
                    qparsed = {}
                if isinstance(qparsed, dict):
                    queue_payload = qparsed
            meta = resolve_video_status(
                prompt_id,
                local=local,
                history=hist_payload,
                queue=queue_payload,
            )
            if meta.get("status") == "completed" and not archive_mp4_path(prompt_id).is_file():
                blob = fetch_comfy_asset(comfy_base, meta.get("asset") or {})
                if blob:
                    meta = write_archive(meta, blob)
            with lock:
                merged = apply_resolved_status(jobs.get(prompt_id), meta)
                jobs[prompt_id] = merged
                return dict(merged)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            if path in {"/health", "/v1/health"}:
                alive = comfy_proc.poll() is None
                self._send(200 if alive else 503, {"ok": alive, "engine": "comfyui"})
                return
            if path in {"/metrics"}:
                self._send(200, b"# TYPE up gauge\nup 1\n", "text/plain; version=0.0.4")
                return
            if path in {"/v1/models", "/models"}:
                self._send(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": served_name,
                                "object": "model",
                                "owned_by": "comfyui",
                                "permission": [],
                                "capabilities": H3_CAPABILITIES,
                            }
                        ],
                    },
                )
                return
            if path in {"/v1/videos", "/v1/video/generations"}:
                with lock:
                    rows = [
                        {
                            "id": row.get("id"),
                            "object": "video",
                            "status": row.get("status"),
                            "prompt": row.get("prompt") or "",
                            "seconds": row.get("seconds"),
                            "size": row.get("size"),
                            "submitted_at": row.get("submitted_at"),
                            "url": row.get("url") or f"/v1/videos/{row.get('id')}/content",
                            "segment": row.get("segment"),
                            "segments": row.get("segments"),
                        }
                        for row in jobs.values()
                        if row.get("id") and str(row.get("status") or "") in {
                            "completed", "in_progress", "queued",
                        }
                    ]
                rows.sort(key=lambda r: float(r.get("submitted_at") or 0), reverse=True)
                self._send(200, {"object": "list", "data": rows[:48]})
                return
            if path in {"/v1/videos/engine"}:
                self._send(
                    200,
                    {
                        "object": "video.engine",
                        "model": served_name,
                        "available": comfy_proc.poll() is None,
                        "capabilities": H3_CAPABILITIES,
                    },
                )
                return
            if path in {"/system_stats"}:
                code, payload, ctype = _http_json(f"{comfy_base}/system_stats", timeout=5)
                self._send(code, payload, ctype)
                return
            if path.startswith("/v1/videos/") and path.endswith("/content"):
                prompt_id = path[len("/v1/videos/") : -len("/content")]
                archived = archive_mp4_path(prompt_id)
                if archived.is_file():
                    self._send(200, archived.read_bytes(), "video/mp4")
                    return
                meta = self._video_status(prompt_id)
                asset = meta.get("asset") or {}
                if meta.get("status") != "completed" or not asset.get("filename"):
                    self._send(404, {"error": "video not ready", "status": meta.get("status")})
                    return
                qs = (
                    f"filename={asset['filename']}"
                    f"&subfolder={asset.get('subfolder') or ''}"
                    f"&type={asset.get('type') or 'output'}"
                )
                try:
                    req = Request(f"{comfy_base}/view?{qs}")
                    with urlopen(req, timeout=60) as resp:
                        blob = resp.read()
                        ctype = resp.headers.get("content-type") or "video/mp4"
                    self._send(200, blob, ctype)
                except (URLError, TimeoutError, OSError) as exc:
                    self._send(502, {"error": str(exc)})
                return
            if path.startswith("/v1/videos/"):
                prompt_id = path[len("/v1/videos/") :]
                self._send(200, self._video_status(prompt_id))
                return
            self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path.rstrip("/") or "/"
            length = int(self.headers.get("content-length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            if path in {"/v1/videos", "/v1/video/generations"}:
                try:
                    body = json.loads(raw.decode() or "{}")
                except json.JSONDecodeError:
                    self._send(400, {"error": "invalid json"})
                    return
                prompt = str(body.get("prompt") or "").strip()
                if not prompt:
                    self._send(400, {"error": "prompt is required"})
                    return
                width, height = parse_size(
                    str(body.get("size") or ""),
                    resolution=str(body.get("resolution") or ""),
                    aspect=str(body.get("aspect") or ""),
                )
                if body.get("width") and body.get("height"):
                    width, height = parse_size(f"{body['width']}x{body['height']}")
                turbo = body.get("turbo")
                if turbo is None:
                    turbo = True
                target = parse_target_seconds(body.get("seconds"), 5.0)
                shot = max_seconds_for_size(width, height)
                first_name = last_name = None
                try:
                    first_blob = _frame_from_body(
                        body, "first_frame", "start_image", "input_reference", "image",
                    )
                    last_blob = _frame_from_body(body, "last_frame", "end_image")
                    if first_blob:
                        first_name = _comfy_upload_image(comfy_base, first_blob[0], first_blob[1])
                    if last_blob:
                        last_name = _comfy_upload_image(comfy_base, last_blob[0], last_blob[1])
                except (URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError) as exc:
                    self._send(502, {"error": f"could not stage input frame: {exc}"})
                    return
                seed = int(body.get("seed") or 1)
                if target > shot + 0.01:
                    job_id = str(uuid.uuid4())
                    meta = {
                        "id": job_id,
                        "object": "video",
                        "model": served_name,
                        "status": "queued",
                        "prompt": prompt,
                        "seconds": target,
                        "size": f"{width}x{height}",
                        "turbo": bool(turbo),
                        "compose": True,
                        "segments": int(math.ceil(target / shot)),
                        "submitted_at": time.time(),
                    }
                    with lock:
                        jobs[job_id] = meta
                    threading.Thread(
                        target=run_compose_job,
                        kwargs={
                            "job_id": job_id,
                            "comfy_base": comfy_base,
                            "jobs": jobs,
                            "lock": lock,
                            "prompt": prompt,
                            "width": width,
                            "height": height,
                            "target_s": target,
                            "shot_s": shot,
                            "seed": seed,
                            "turbo": bool(turbo),
                            "first_name": first_name,
                        },
                        daemon=True,
                    ).start()
                    self._send(200, meta)
                    return
                seconds = parse_seconds(target, 5.0, width=width, height=height)
                graph = t2v_api_prompt(
                    prompt,
                    width=width,
                    height=height,
                    duration_s=seconds,
                    seed=seed,
                    turbo=bool(turbo),
                    steps=body.get("steps"),
                    first_image_name=first_name,
                    last_image_name=last_name,
                )
                code, payload, _ = _http_json(
                    f"{comfy_base}/prompt",
                    data=json.dumps({"prompt": graph}).encode(),
                    timeout=30,
                )
                try:
                    parsed = json.loads(payload.decode() or "{}")
                except json.JSONDecodeError:
                    parsed = {"error": payload.decode("utf-8", "replace")}
                if code >= 400 or not parsed.get("prompt_id"):
                    self._send(code if code >= 400 else 502, parsed)
                    return
                prompt_id = str(parsed["prompt_id"])
                meta = {
                    "id": prompt_id,
                    "object": "video",
                    "model": served_name,
                    "status": "queued",
                    "prompt": prompt,
                    "seconds": seconds,
                    "size": f"{width}x{height}",
                    "turbo": bool(turbo),
                    "has_first_frame": bool(first_name),
                    "has_last_frame": bool(last_name),
                    "submitted_at": time.time(),
                }
                with lock:
                    jobs[prompt_id] = meta
                self._send(200, meta)
                return
            if path == "/prompt":
                code, payload, ctype = _http_json(
                    f"{comfy_base}/prompt", data=raw, timeout=30
                )
                self._send(code, payload, ctype)
                return
            self._send(404, {"error": "not found"})

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cortex-deployer-comfyui")
    parser.add_argument("--comfy-root", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--served-name", default="MiniMax-H3")
    parser.add_argument(
        "--comfy-port",
        type=int,
        default=0,
        help="ComfyUI listen port (default: public+1)",
    )
    args, extra = parser.parse_known_args(argv)
    comfy_root = Path(args.comfy_root).expanduser()
    comfy_port = int(args.comfy_port) or (int(args.port) + 1)
    proc = _spawn_comfy(comfy_root, "127.0.0.1", comfy_port, extra)
    stats = f"{_comfy_base('127.0.0.1', comfy_port)}/system_stats"
    httpd_holder: list[ThreadingHTTPServer] = []

    def _stop(_signum=None, _frame=None) -> None:
        _stop_comfy(proc)
        if httpd_holder:
            httpd_holder[0].shutdown()
        else:
            raise SystemExit(0)

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    try:
        _wait_comfy(stats)
    except Exception:
        _stop_comfy(proc)
        raise
    handler = _handler(args.served_name, _comfy_base("127.0.0.1", comfy_port), proc, comfy_root)
    httpd = ThreadingHTTPServer((args.host, int(args.port)), handler)
    httpd_holder.append(httpd)

    def _reap() -> None:
        proc.wait()
        httpd.shutdown()

    threading.Thread(target=_reap, daemon=True).start()
    try:
        httpd.serve_forever()
    finally:
        _stop_comfy(proc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
