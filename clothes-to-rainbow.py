import argparse
import os
import numpy as np
import torch
from PIL import Image, ImageFilter
from dotenv import load_dotenv
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from novelai import NovelAI
from novelai.types import GenerateImageParams, InpaintParams

load_dotenv()

api_key = os.getenv("NOVELAI_API_KEY")
if not api_key:
    raise RuntimeError("NOVELAI_API_KEY is not set")

MODEL_NAME_CLOTH = "mattmdjaga/segformer_b2_clothes"
CLOTH_LABELS = { 4, 5, 6, 7, 8, 9, 10, 17 }

MODEL_NAME_NAI = "nai-diffusion-4-5-full"

def create_mask_path(input_path: str) -> str:
    base, _ = os.path.splitext(input_path)
    return f"{base}-masked.png"


def create_output_path(input_path: str, suffix: str) -> str:
    base, _ = os.path.splitext(input_path)
    return f"{base}-{suffix}.png"


def generate_clothes_mask(input_path: str):
    image = Image.open(input_path).convert("RGB")
    w, h = image.size

    processor = SegformerImageProcessor.from_pretrained(MODEL_NAME_CLOTH)
    model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_NAME_CLOTH)
    model.eval()

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    )

    pred = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
    clothes_mask = np.isin(pred, list(CLOTH_LABELS))

    mask_l = Image.fromarray((clothes_mask.astype(np.uint8) * 255))

    # マスクを一回り大きくする 一旦22 要チューニング
    expand_pixels = 22
    mask_l = mask_l.filter(ImageFilter.MaxFilter(expand_pixels * 2 + 1))

    alpha = np.array(mask_l)

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 3] = alpha
    rgba[:, :, 0] = 0
    rgba[:, :, 1] = 0
    rgba[:, :, 2] = 0

    mask_path = create_mask_path(input_path)
    Image.fromarray(rgba).save(mask_path, format="PNG")

    return mask_path


def mask_to_rainbow(input_path: str, mask_path: str):

    client = NovelAI(api_key=api_key)

    output_path = create_output_path(input_path, "rainbow")

    inpaint_params = InpaintParams(
        image=input_path,
        mask=mask_path,
        strength=1.0,
    )

    with Image.open(input_path) as img:
        size_w, size_h = img.size
        size_w = min(4096, max(256, round(size_w / 64) * 64))
        size_h = min(4096, max(256, round(size_h / 64) * 64))

    params = GenerateImageParams(
        prompt="rainbow dress, rainbow clothes, masterpiece, best quality",
        model=MODEL_NAME_NAI,
        inpaint=inpaint_params,
        size=(size_w, size_h),
    )

    images = client.image.generate(params)
    images[0].save(output_path)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input image (jpg/jpeg/png)")
    args = parser.parse_args()

    mask_path = generate_clothes_mask(args.input)
    mask_to_rainbow(args.input, mask_path)


if __name__ == "__main__":
    main()