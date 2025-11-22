import os
import io
import base64
import zipfile
import shutil
from pathlib import Path

import requests
import runpod
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

# ------------------------------
# Model setup
# ------------------------------

MODEL_ID = os.getenv("MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIPELINE = None


def load_pipeline():
    """Lazy-load SDXL pipeline once per container."""
    global PIPELINE
    if PIPELINE is not None:
        return PIPELINE

    print(f"[init] Loading SDXL model: {MODEL_ID} on {DEVICE}")
    PIPELINE = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16
    ).to(DEVICE)

    try:
        PIPELINE.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    return PIPELINE


def image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ------------------------------
# Dataset helpers for train-lora
# ------------------------------

def prepare_dataset(zip_url: str, avatar_id: str | None = None):
    """
    Download the dataset ZIP from zip_url and extract it under /tmp.
    Returns (dataset_dir, image_files).
    """
    if not zip_url:
        raise ValueError("zip_url is required")

    suffix = avatar_id or "default"
    base_dir = Path(f"/tmp/dataset_{suffix}")

    # Clean old dir if it exists
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    zip_path = base_dir / "dataset.zip"

    print(f"[train] Downloading dataset from {zip_url}")
    with requests.get(zip_url, stream=True) as resp:
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    print(f"[train] Extracting dataset to {base_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(base_dir)

    # Collect image files
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    image_files = [p for p in base_dir.rglob("*") if p.suffix.lower() in exts]

    print(f"[train] Found {len(image_files)} image files")

    return base_dir, image_files


def create_dummy_lora(dataset_dir: Path, avatar_id: str | None = None) -> Path:
    """
    TEMP STEP: create a dummy .safetensors file to test the pipeline.
    Later we will replace this with REAL LoRA training.
    """
    name = avatar_id or "avatar"
    out_path = Path(f"/tmp/{name}_dummy_lora.safetensors")

    print(f"[train] Creating dummy LoRA file at {out_path}")
    # Just write some bytes so the file exists
    with open(out_path, "wb") as f:
        f.write(b"FAKE_LORA_DATA")

    return out_path


def upload_lora_file(local_path: Path, upload_url: str | None):
    """
    Upload a local LoRA file to the provided signed URL (if any).
    """
    if not upload_url:
        print("[train] No lora_upload_url provided, skipping upload.")
        return

    if not local_path.exists():
        raise FileNotFoundError(f"LoRA file not found at {local_path}")

    print(f"[train] Uploading LoRA to {upload_url}")
    with open(local_path, "rb") as f:
        resp = requests.put(upload_url, data=f)
        resp.raise_for_status()
    print("[train] LoRA upload completed")


# ------------------------------
# RunPod handler
# ------------------------------

def handler(job):
    """
    job['input'] can be:

    1) Generate:
    {
      "task": "generate",
      "engine": "sdxl",
      "prompt": "...",
      "negative_prompt": "...",
      "num_outputs": 1,
      "width": 1024,
      "height": 1024,
      "steps": 30,
      "cfg_scale": 7,
      "seed": 12345,
      "lora_url": "https://..."   # optional
    }

    2) Train LoRA (dummy training for now):
    {
      "task": "train-lora",
      "engine": "sdxl",
      "user_id": "user_123",
      "avatar_id": "avatar_abc",
      "zip_url": "https://.../dataset.zip",
      "trigger_word": "Sofia",
      "steps": 2000,
      "lora_upload_url": "https://.../PUT.safetensors",
      "lora_public_url": "https://.../GET.safetensors"
    }
    """
    data = job.get("input", {}) or {}
    task = data.get("task", "generate")

    # --------------------------
    # TRAIN LORA (dummy version)
    # --------------------------
    if task == "train-lora":
        user_id = data.get("user_id")
        avatar_id = data.get("avatar_id")
        zip_url = data.get("zip_url")
        trigger_word = data.get("trigger_word")
        steps = data.get("steps")
        lora_upload_url = data.get("lora_upload_url")
        lora_public_url = data.get("lora_public_url")

        try:
            dataset_dir, image_files = prepare_dataset(zip_url, avatar_id)

            # TEMP: create dummy LoRA file instead of real training
            local_lora_path = create_dummy_lora(dataset_dir, avatar_id)

            # TEMP: upload dummy LoRA (if URL provided)
            try:
                upload_lora_file(local_lora_path, lora_upload_url)
                upload_status = "uploaded"
            except Exception as up_err:
                upload_status = f"upload_failed: {up_err}"

            return {
                "status": "trained_stub",  # will change to 'done' when real training is in place
                "engine": "sdxl",
                "user_id": user_id,
                "avatar_id": avatar_id,
                "zip_url": zip_url,
                "trigger_word": trigger_word,
                "steps": steps,
                "lora_upload_url": lora_upload_url,
                "lora_public_url": lora_public_url,
                "dataset_dir": str(dataset_dir),
                "num_images": len(image_files),
                "local_lora_path": str(local_lora_path),
                "upload_status": upload_status,
            }
        except Exception as e:
            return {
                "status": "error_training_stub",
                "error_message": str(e),
                "engine": "sdxl",
                "user_id": user_id,
                "avatar_id": avatar_id,
                "zip_url": zip_url,
            }

    # --------------------------
    # GENERATE
    # --------------------------
    elif task == "generate":
        prompt = data.get("prompt", "a photo of a woman, studio portrait, 4k")
        negative_prompt = data.get(
            "negative_prompt",
            "extra limbs, extra fingers, bad anatomy, deformed, blurry, low quality"
        )

        num_outputs = int(data.get("num_outputs", 1))
        width = int(data.get("width", 1024))
        height = int(data.get("height", 1024))
        steps = int(data.get("steps", 30))
        cfg_scale = float(data.get("cfg_scale", 7.0))
        seed = data.get("seed", None)
        lora_url = data.get("lora_url")

        generator = None
        if seed is not None:
            generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

        pipe = load_pipeline()

        # Optional LoRA loading
        unload_after = False
        if lora_url:
            try:
                resp = requests.get(lora_url)
                resp.raise_for_status()
                lora_path = "/tmp/avatar_lora.safetensors"
                with open(lora_path, "wb") as f:
                    f.write(resp.content)

                pipe.load_lora_weights(lora_path)
                unload_after = True
                print(f"[lora] Loaded LoRA from {lora_url}")
            except Exception as e:
                print(f"[lora] Failed to load LoRA from {lora_url}: {e}")

        with torch.autocast("cuda"):
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                width=width,
                height=height,
                num_images_per_prompt=num_outputs,
                generator=generator,
            )

        images = result.images

        encoded = [
            {
                "index": i,
                "format": "png",
                "base64": image_to_base64(img, "PNG"),
            }
            for i, img in enumerate(images)
        ]

        if unload_after:
            try:
                pipe.unload_lora_weights()
            except Exception:
                pass

        return {
            "status": "ok",
            "engine": "sdxl",
            "images": encoded,
        }

    # --------------------------
    # UNKNOWN TASK
    # --------------------------
    else:
        return {"error": f"Unsupported task: {task}"}


runpod.serverless.start({"handler": handler})
