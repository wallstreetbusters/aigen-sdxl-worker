import requests

import os
import io
import base64

import runpod
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

# ------------------------------
# Model setup
# ------------------------------

# You can override this in RunPod env vars later if you want a different SDXL model
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

    # If xformers is available, this saves VRAM
    try:
        PIPELINE.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    return PIPELINE


def image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL image as base64 string."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ------------------------------
# RunPod handler
# ------------------------------

def handler(job):
    """
    RunPod serverless handler.

    Expected input format (job['input']):

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
      "seed": 12345
    }
    """
    data = job.get("input", {}) or {}

    task = data.get("task", "generate")
    if task != "generate":
        return {"error": f"Unsupported task: {task}"}

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
    lora_url = data.get("lora_url")  # URL to a .safetensors LoRA file (R2 later)


    generator = None
    if seed is not None:
        generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

    pipe = load_pipeline()

    # Optionally load a LoRA for avatar generation
unload_after = False
if lora_url:
    try:
        # Download LoRA file from URL to a temp path
        resp = requests.get(lora_url)
        resp.raise_for_status()
        lora_path = "/tmp/avatar_lora.safetensors"
        with open(lora_path, "wb") as f:
            f.write(resp.content)

        # Load LoRA weights into the pipeline
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


# Required by RunPod
runpod.serverless.start({"handler": handler})
