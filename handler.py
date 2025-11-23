import os
import io
import base64
import zipfile
import shutil
from pathlib import Path
from typing import List
import traceback  # for logging errors

import numpy as np
import requests
import runpod
import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionXLPipeline, DDPMScheduler
from peft import LoraConfig
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
# Simple dataset for LoRA training
# ------------------------------

class AvatarDataset(Dataset):
    """
    Very small in-memory dataset:
    - image_files: list of Paths
    - caption: single caption for all images (e.g. "a photo of Sofia")
    """
    def __init__(self, image_files: List[Path], caption: str, resolution: int = 768):
        self.image_files = image_files
        self.caption = caption
        self.resolution = resolution

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        path = self.image_files[idx]

        # Load and clean image
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            # Fallback to first image if there is a problem
            image = Image.open(self.image_files[0]).convert("RGB")

        w, h = image.size
        short = min(w, h)
        left = (w - short) // 2
        top = (h - short) // 2
        image = image.crop((left, top, left + short, top + short))
        image = image.resize((self.resolution, self.resolution), Image.BICUBIC)

        # To tensor in [-1, 1]
        arr = np.array(image).astype(np.float32) / 255.0  # H, W, C in [0,1]
        arr = arr.transpose(2, 0, 1)  # C, H, W
        tensor = torch.from_numpy(arr) * 2.0 - 1.0

        return {
            "pixel_values": tensor,
            "caption": self.caption,
        }


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


LORA_API_KEY = os.getenv("LORA_UPLOAD_API_KEY", "")


def upload_lora_file(local_path: Path, upload_url: str | None):
    """
    Upload a local LoRA file to the lora-uploader Worker.
    We send x-api-key so the Worker can auth us.
    """
    if not upload_url:
        print("[train] No lora_upload_url provided, skipping upload.")
        return

    if not local_path.exists():
        raise FileNotFoundError(f"LoRA file not found at {local_path}")

    headers = {}
    if LORA_API_KEY:
        headers["x-api-key"] = LORA_API_KEY

    print(f"[train] Uploading LoRA to {upload_url}")
    with open(local_path, "rb") as f:
        resp = requests.put(upload_url, data=f, headers=headers)
        resp.raise_for_status()
    print("[train] LoRA upload completed")


    def map_ui_steps_to_train_steps(ui_steps: int, num_images: int) -> int:
    """
    Map frontend 'steps' (2000/3000/4000/5000) + image count (12–18)
    to a sane SDXL LoRA training step count.

    Rough idea:
      - Base ~100 steps per image (community practice: ~75–120× num_images)
      - Higher UI steps => higher factor
      - Clamp to a safe range for SDXL on serverless
    """
    # Safety: clamp num_images to a reasonable range
    num_images = max(1, min(int(num_images), 32))

    base = num_images * 100  # 100 steps per image as baseline

    # Map UI value to a quality factor
    if ui_steps <= 2000:
        factor = 0.8   # lower end of quality
    elif ui_steps <= 3000:
        factor = 1.0   # baseline
    elif ui_steps <= 4000:
        factor = 1.2   # a bit more training
    else:  # 5000 or above
        factor = 1.4   # highest quality preset

    effective = int(base * factor)

    # Final clamp to keep things reasonable for SDXL LoRA
    # (roughly 600–2200 steps for 12–18 images)
    effective = max(600, min(effective, 2200))

    return effective

# ------------------------------
# REAL SDXL LoRA TRAINING
# ------------------------------

def train_lora_sdxl(
    dataset_dir: Path,
    image_files: List[Path],
    avatar_id: str,
    steps: int,
    trigger_word: str | None = None,
) -> Path:
    """
    Simplified SDXL LoRA training loop using pipeline helpers.
    """
    if not image_files:
        raise ValueError("No images found for training.")

    # --- Map UI steps (2000/3000/4000/5000) -> real training steps ---
    try:
        ui_steps = int(steps) if steps is not None else 2000
    except Exception:
        ui_steps = 2000

    num_images = len(image_files)
    effective_steps = map_ui_steps_to_train_steps(ui_steps, num_images)

    print(
        f"[train] avatar_id={avatar_id} num_images={num_images} "
        f"ui_steps={ui_steps} -> effective_steps={effective_steps}"
    )

    steps = effective_steps  # from here on, training uses this value


    # 1) Build caption
    if trigger_word:
        caption = f"a photo of {trigger_word}"
    else:
        caption = f"a photo of {avatar_id}"

    print(f"[train] Starting LoRA training for avatar '{avatar_id}'")
    print(f"[train] Using caption: {caption}")
    print(f"[train] Steps: {steps}")

    # 2) Prepare dataset & dataloader
    print("[train] Building dataset and dataloader")
    dataset = AvatarDataset(image_files=image_files, caption=caption, resolution=768)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # 3) Load SDXL pipeline and components
    pipe = load_pipeline()
    pipe.to(DEVICE)

    unet = pipe.unet
    vae = pipe.vae
    scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    height = width = dataset.resolution

       # 4) Get text + pooled embeddings and time ids via SDXL helpers
    try:
        # We do NOT want gradients through the text encoder for LoRA-on-UNet training,
        # so we wrap encode_prompt + _get_add_time_ids in torch.no_grad().
        with torch.no_grad():
            # SDXL encode_prompt returns 4 values
            (
                prompt_embeds,
                _neg_prompt_embeds,
                pooled_embeds,
                _neg_pooled,
            ) = pipe.encode_prompt(
                caption,
                device=DEVICE,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            print("[train] Got prompt & pooled embeddings from encode_prompt")

            # IMPORTANT: for SDXL we must pass text_encoder_projection_dim explicitly
            text_encoder_projection_dim = None
            if hasattr(pipe, "text_encoder_2") and hasattr(
                pipe.text_encoder_2, "config"
            ):
                text_encoder_projection_dim = getattr(
                    pipe.text_encoder_2.config, "projection_dim", None
                )
            if (
                text_encoder_projection_dim is None
                and hasattr(pipe, "text_encoder")
                and hasattr(pipe.text_encoder, "config")
            ):
                text_encoder_projection_dim = getattr(
                    pipe.text_encoder.config, "projection_dim", None
                )

            if text_encoder_projection_dim is None:
                raise RuntimeError(
                    "Could not determine text_encoder_projection_dim for SDXL."
                )

            add_time_ids = pipe._get_add_time_ids(
                (height, width),
                (0, 0),
                (height, width),
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
            print("[train] Got time ids from _get_add_time_ids")

        # Make sure these are on the right device/dtype and detached (no grad graph)
        prompt_embeds = prompt_embeds.to(DEVICE, dtype=torch.float16)
        pooled_embeds = pooled_embeds.to(DEVICE, dtype=torch.float16)
        add_time_ids = add_time_ids.to(DEVICE, dtype=torch.float16)

    except Exception as e:
        print(f"[train] FATAL: encode_prompt/_get_add_time_ids failed: {e}")
        raise



    prompt_embeds = prompt_embeds.to(DEVICE)
    pooled_embeds = pooled_embeds.to(DEVICE)
    add_time_ids = add_time_ids.to(DEVICE)

    # 5) Freeze everything except LoRA params
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    if hasattr(pipe, "text_encoder"):
        pipe.text_encoder.requires_grad_(False)
    if hasattr(pipe, "text_encoder_2"):
        pipe.text_encoder_2.requires_grad_(False)

    # 6) Attach LoRA to UNet (once per process)
    adapter_name = "default"

    if getattr(unet, "peft_config", None) is None:
        unet.peft_config = {}

    if adapter_name in unet.peft_config:
        print(f"[train] LoRA adapter '{adapter_name}' already exists, reusing it.")
    else:
        unet_lora_config = LoraConfig(
            r=8,
            lora_alpha=8,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        print(f"[train] Adding LoRA adapter '{adapter_name}' to UNet")
        unet.add_adapter(unet_lora_config, adapter_name)

    # Only LoRA params will have requires_grad=True
    lora_params = [p for p in unet.parameters() if p.requires_grad]
    print(f"[train] Number of trainable LoRA params: {sum(p.numel() for p in lora_params)}")

    optimizer = torch.optim.AdamW(lora_params, lr=1e-5)

    # 7) Training loop
    global_step = 0
    pipe.unet.train()

    noise_scheduler = scheduler

    print("[train] Entering training loop")
    while global_step < steps:
        for batch in dataloader:
            if global_step >= steps:
                break

            pixel_values = batch["pixel_values"].to(DEVICE, dtype=torch.float16)

            # Encode images to latents
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * 0.18215

            # Sample noise and timesteps
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=latents.device,
                dtype=torch.long,
            )

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Build encoder_hidden_states and added_cond_kwargs for SDXL
            bsz = latents.shape[0]
            encoder_hidden_states = prompt_embeds.repeat(bsz, 1, 1)
            text_embeds = pooled_embeds.repeat(bsz, 1)
            time_ids = add_time_ids.repeat(bsz, 1)

            added_cond_kwargs = {
                "text_embeds": text_embeds,
                "time_ids": time_ids,
            }

            # 8) Forward UNet with correct SDXL conditioning
with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
    model_pred = unet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=encoder_hidden_states,
        added_cond_kwargs=added_cond_kwargs,
    ).sample

    loss = torch.nn.functional.mse_loss(model_pred, noise)

# Guard against NaN / Inf loss
if not torch.isfinite(loss):
    print(f"[train] WARNING: non-finite loss at step {global_step}: {loss.item()}")
    global_step += 1
    if global_step % 10 == 0:
        print(f"[train] Step {global_step}/{steps} - loss is non-finite, skipping")
    if global_step >= steps:
        break
    continue

optimizer.zero_grad()
loss.backward()

# Gradient clipping to avoid exploding grads
torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)

optimizer.step()

global_step += 1

if global_step % 10 == 0:
    print(f"[train] Step {global_step}/{steps} - loss: {loss.item():.4f}")


    # 9) Save LoRA weights to a temp directory, then move to a single safetensors file
    tmp_out_dir = Path(f"/tmp/{avatar_id}_lora_out")
    if tmp_out_dir.exists():
        shutil.rmtree(tmp_out_dir)
    tmp_out_dir.mkdir(parents=True, exist_ok=True)

    # Uses diffusers' built-in LoRA saving
    pipe.save_lora_weights(tmp_out_dir.as_posix(), adapter_name=adapter_name)

    # Look for a safetensors file
    lora_file = None
    for p in tmp_out_dir.glob("*.safetensors"):
        lora_file = p
        break

    if lora_file is None:
        raise FileNotFoundError(f"No .safetensors LoRA file found in {tmp_out_dir}")

    final_path = Path(f"/tmp/{avatar_id}_trained_lora.safetensors")
    shutil.copy2(lora_file, final_path)

    print(f"[train] LoRA saved to {final_path}")
    return final_path


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

    2) Train LoRA:
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
    # TRAIN LORA (real version)
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

            # REAL SDXL LoRA training
            steps_int = int(steps) if steps is not None else 800
            local_lora_path = train_lora_sdxl(
                dataset_dir=dataset_dir,
                image_files=image_files,
                avatar_id=avatar_id or "avatar",
                steps=steps_int,
                trigger_word=trigger_word,
            )

            # Upload trained LoRA (if URL provided)
            try:
                upload_lora_file(local_lora_path, lora_upload_url)
                upload_status = "uploaded"
            except Exception as up_err:
                upload_status = f"upload_failed: {up_err}"

            return {
                "status": "trained_sd_lora",
                "engine": "sdxl",
                "user_id": user_id,
                "avatar_id": avatar_id,
                "zip_url": zip_url,
                "trigger_word": trigger_word,
                "steps": steps_int,
                "lora_upload_url": lora_upload_url,
                "lora_public_url": lora_public_url,
                "dataset_dir": str(dataset_dir),
                "num_images": len(image_files),
                "local_lora_path": str(local_lora_path),
                "upload_status": upload_status,
            }
        except Exception as e:
            print("[train] ERROR in train-lora:")
            traceback.print_exc()
            return {
                "status": "error_training_sdxl",
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

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
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
