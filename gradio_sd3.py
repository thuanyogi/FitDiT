import gradio as gr
import os
import math
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.dwpose import DWposeDetector
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
import torch
import torch.nn as nn
from src.pose_guider import PoseGuider
from PIL import Image
from src.utils_mask import get_mask_location
import numpy as np
from src.pipeline_stable_diffusion_3_tryon import StableDiffusion3TryOnPipeline
from src.transformer_sd3_garm import SD3Transformer2DModel as SD3Transformer2DModel_Garm
from src.transformer_sd3_vton import SD3Transformer2DModel as SD3Transformer2DModel_Vton
import cv2
import random

example_path = os.path.join(os.path.dirname(__file__), 'examples')


class FitDiTGenerator:
    def __init__(self, model_root, offload=False, aggressive_offload=False, device="cuda:0", with_fp16=False):
        weight_dtype = torch.float16 if with_fp16 else torch.bfloat16
        transformer_garm = SD3Transformer2DModel_Garm.from_pretrained(os.path.join(model_root, "transformer_garm"), torch_dtype=weight_dtype)
        transformer_vton = SD3Transformer2DModel_Vton.from_pretrained(os.path.join(model_root, "transformer_vton"), torch_dtype=weight_dtype)
        pose_guider =  PoseGuider(conditioning_embedding_channels=1536, conditioning_channels=3, block_out_channels=(32, 64, 256, 512))
        pose_guider.load_state_dict(torch.load(os.path.join(model_root, "pose_guider", "diffusion_pytorch_model.bin")))
        image_encoder_large = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=weight_dtype)
        image_encoder_bigG = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", torch_dtype=weight_dtype)
        pose_guider.to(device=device, dtype=weight_dtype)
        image_encoder_large.to(device=device)
        image_encoder_bigG.to(device=device)
        self.pipeline = StableDiffusion3TryOnPipeline.from_pretrained(model_root, torch_dtype=weight_dtype, transformer_garm=transformer_garm, transformer_vton=transformer_vton, pose_guider=pose_guider, image_encoder_large=image_encoder_large, image_encoder_bigG=image_encoder_bigG)
        self.pipeline.to(device)
        if offload:
            self.pipeline.enable_model_cpu_offload()
            self.dwprocessor = DWposeDetector(model_root=model_root, device='cpu')
            self.parsing_model = Parsing(model_root=model_root, device='cpu')
        elif aggressive_offload:
            self.pipeline.enable_sequential_cpu_offload()
            self.dwprocessor = DWposeDetector(model_root=model_root, device='cpu')
            self.parsing_model = Parsing(model_root=model_root, device='cpu')
        else:
            self.pipeline.to(device)
            self.dwprocessor = DWposeDetector(model_root=model_root, device=device)
            self.parsing_model = Parsing(model_root=model_root, device=device)
        
    def generate_mask(self, vton_img, category, offset_top, offset_bottom, offset_left, offset_right):
        with torch.inference_mode():
            vton_img = Image.open(vton_img)
            vton_img_det = resize_image(vton_img)
            pose_image, keypoints, _, candidate = self.dwprocessor(np.array(vton_img_det)[:,:,::-1])
            candidate[candidate<0]=0
            candidate = candidate[0]

            candidate[:, 0]*=vton_img_det.width
            candidate[:, 1]*=vton_img_det.height

            pose_image = pose_image[:,:,::-1] #rgb
            pose_image = Image.fromarray(pose_image)
            model_parse, _ = self.parsing_model(vton_img_det)

            mask, mask_gray = get_mask_location(category, model_parse, \
                                        candidate, model_parse.width, model_parse.height, \
                                        offset_top, offset_bottom, offset_left, offset_right)
            mask = mask.resize(vton_img.size)
            mask_gray = mask_gray.resize(vton_img.size)
            mask = mask.convert("L")
            mask_gray = mask_gray.convert("L")
            masked_vton_img = Image.composite(mask_gray, vton_img, mask)

            im = {}
            im['background'] = np.array(vton_img.convert("RGBA"))
            im['layers'] = [np.concatenate((np.array(mask_gray.convert("RGB")), np.array(mask)[:,:,np.newaxis]),axis=2)]
            im['composite'] = np.array(masked_vton_img.convert("RGBA"))
            
            return im, pose_image

    def process(self, vton_img, garm_img, pre_mask, pose_image, n_steps, image_scale, seed, num_images_per_prompt, resolution):
        assert resolution in ["768x1024", "1152x1536", "1536x2048"]
        new_width, new_height = resolution.split("x")
        new_width = int(new_width)
        new_height = int(new_height)
        with torch.inference_mode():
            garm_img = Image.open(garm_img)
            vton_img = Image.open(vton_img)

            model_image_size = vton_img.size
            garm_img, _, _ = pad_and_resize(garm_img, new_width=new_width, new_height=new_height)
            vton_img, pad_w, pad_h = pad_and_resize(vton_img, new_width=new_width, new_height=new_height)

            mask = pre_mask["layers"][0][:,:,3]
            mask = Image.fromarray(mask)
            mask, _, _ = pad_and_resize(mask, new_width=new_width, new_height=new_height, pad_color=(0,0,0))
            mask = mask.convert("L")
            pose_image = Image.fromarray(pose_image)
            pose_image, _, _ = pad_and_resize(pose_image, new_width=new_width, new_height=new_height, pad_color=(0,0,0))
            if seed==-1:
                seed = random.randint(0, 2147483647)
            res = self.pipeline(
                height=new_height,
                width=new_width,
                guidance_scale=image_scale,
                num_inference_steps=n_steps,
                generator=torch.Generator("cpu").manual_seed(seed),
                cloth_image=garm_img,
                model_image=vton_img,
                mask=mask,
                pose_image=pose_image,
                num_images_per_prompt=num_images_per_prompt
            ).images
            for idx in range(len(res)):
                res[idx] = unpad_and_resize(res[idx], pad_w, pad_h, model_image_size[0], model_image_size[1])
            return res


def pad_and_resize(im, new_width=768, new_height=1024, pad_color=(255, 255, 255), mode=Image.LANCZOS):
    old_width, old_height = im.size
    
    ratio_w = new_width / old_width
    ratio_h = new_height / old_height
    if ratio_w < ratio_h:
        new_size = (new_width, round(old_height * ratio_w))
    else:
        new_size = (round(old_width * ratio_h), new_height)
    
    im_resized = im.resize(new_size, mode)

    pad_w = math.ceil((new_width - im_resized.width) / 2)
    pad_h = math.ceil((new_height - im_resized.height) / 2)

    new_im = Image.new('RGB', (new_width, new_height), pad_color)
    
    new_im.paste(im_resized, (pad_w, pad_h))

    return new_im, pad_w, pad_h

def unpad_and_resize(padded_im, pad_w, pad_h, original_width, original_height):
    width, height = padded_im.size
    
    left = pad_w
    top = pad_h
    right = width - pad_w
    bottom = height - pad_h
    
    cropped_im = padded_im.crop((left, top, right, bottom))

    resized_im = cropped_im.resize((original_width, original_height), Image.LANCZOS)

    return resized_im

def resize_image(img, target_size=768):
    width, height = img.size
    
    if width < height:
        scale = target_size / width
    else:
        scale = target_size / height
    
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
    
    return resized_img

CUSTOM_CSS = """
<style>
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --accent-color: #ec4899;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --bg-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    .header-container {
        background: var(--bg-gradient);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .header-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }

    .header-subtitle {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 300;
    }

    .badge-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
        margin-bottom: 1rem;
    }

    .info-box {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        font-size: 0.9rem;
    }

    .step-indicator {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 1.5rem 0;
        gap: 1rem;
    }

    .step {
        background: #f3f4f6;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        color: #6b7280;
        transition: all 0.3s ease;
    }

    .step.active {
        background: var(--primary-color);
        color: white;
        transform: scale(1.05);
    }

    .help-text {
        color: #6b7280;
        font-size: 0.875rem;
        margin-top: 0.25rem;
        font-style: italic;
    }

    .preset-button {
        padding: 0.5rem 1rem;
        border-radius: 6px;
        border: 2px solid var(--primary-color);
        background: white;
        color: var(--primary-color);
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .preset-button:hover {
        background: var(--primary-color);
        color: white;
    }

    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--primary-color);
    }

    .guide-card {
        background: #f9fafb;
        border-left: 4px solid var(--primary-color);
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }

    .guide-step {
        font-weight: 600;
        color: var(--primary-color);
        margin-bottom: 0.5rem;
    }
</style>
"""

HEADER = """
<div class="header-container">
    <h1 class="header-title">✨ FitDiT - Virtual Try-On Studio</h1>
    <p class="header-subtitle">High-fidelity AI-powered garment fitting using Diffusion Transformers</p>
    <div class="badge-container">
        <a href="https://github.com/BoyuanJiang/FitDiT" style="margin: 0 2px;">
            <img src='https://img.shields.io/badge/GitHub-Repo-blue?style=flat&logo=GitHub' alt='GitHub'>
        </a>
        <a href="https://arxiv.org/abs/2411.10499" style="margin: 0 2px;">
            <img src='https://img.shields.io/badge/arXiv-2411.10499-red?style=flat&logo=arXiv&logoColor=red' alt='arxiv'>
        </a>
        <a href="http://demo.fitdit.byjiang.com/" style="margin: 0 2px;">
            <img src='https://img.shields.io/badge/Demo-Gradio-gold?style=flat&logo=Gradio&logoColor=red' alt='Demo'>
        </a>
        <a href='https://byjiang.com/FitDiT/' style="margin: 0 2px;">
            <img src='https://img.shields.io/badge/Webpage-Project-silver?style=flat&logo=&logoColor=orange' alt='webpage'>
        </a>
        <a href="https://raw.githubusercontent.com/BoyuanJiang/FitDiT/refs/heads/main/LICENSE" style="margin: 0 2px;">
            <img src='https://img.shields.io/badge/License-CC BY--NC--SA--4.0-lightgreen?style=flat&logo=Lisence' alt='License'>
        </a>
    </div>
    <div class="info-box">
        <b>⚡ Quick Start:</b> Upload your model photo → Select garment → Adjust mask → Generate results!<br>
        <small>For Non-commercial Use Only | ⭐ <a href="https://github.com/BoyuanJiang/FitDiT" style="color: #fbbf24; text-decoration: underline;">Star us on GitHub</a></small>
    </div>
</div>
"""

GUIDE_CONTENT = """
<div class="guide-card">
    <div class="guide-step">📖 How to Use FitDiT</div>
    <ol style="margin: 0.5rem 0; padding-left: 1.5rem;">
        <li><b>Upload Images:</b> Choose a model photo and a garment image, or select from our examples</li>
        <li><b>Select Category:</b> Pick the garment type (Upper-body, Lower-body, or Dresses)</li>
        <li><b>Generate Mask:</b> Click "Step 1: Run Mask" to detect the area to replace</li>
        <li><b>Adjust Mask (Optional):</b> Fine-tune the mask using offset sliders or draw directly on the image</li>
        <li><b>Configure Settings:</b> Choose quality presets or customize advanced parameters</li>
        <li><b>Generate Result:</b> Click "Step 2: Run Try-on" and wait for your amazing result!</li>
    </ol>
</div>

<div class="guide-card">
    <div class="guide-step">💡 Tips for Best Results</div>
    <ul style="margin: 0.5rem 0; padding-left: 1.5rem;">
        <li>Use clear, well-lit photos with the person facing forward</li>
        <li>Garment images should have clean backgrounds</li>
        <li>Higher resolution settings (1536x2048) give better quality but take longer</li>
        <li>Adjust mask offsets if the initial mask doesn't fit perfectly</li>
        <li>Try different seeds to get varied results</li>
    </ul>
</div>

<div class="guide-card">
    <div class="guide-step">⚙️ Parameter Guide</div>
    <ul style="margin: 0.5rem 0; padding-left: 1.5rem;">
        <li><b>Steps (15-30):</b> More steps = better quality but slower. Start with 20.</li>
        <li><b>Guidance Scale (1-5):</b> Controls how closely the result follows your input. 2-3 is ideal.</li>
        <li><b>Seed:</b> Use -1 for random results, or set a specific number for reproducible results.</li>
        <li><b>Resolution:</b> 768x1024 (Fast), 1152x1536 (Balanced), 1536x2048 (Best Quality)</li>
    </ul>
</div>
"""

def create_demo(model_path, device, offload, aggressive_offload, with_fp16):
    generator = FitDiTGenerator(model_path, offload, aggressive_offload, device, with_fp16)

    # Preset configurations
    def apply_preset(preset_name):
        if preset_name == "⚡ Fast":
            return 15, 2.0, "768x1024"
        elif preset_name == "⚖️ Balanced":
            return 20, 2.5, "1152x1536"
        elif preset_name == "💎 Best Quality":
            return 30, 3.0, "1536x2048"
        return 20, 2.0, "1152x1536"

    with gr.Blocks(title="FitDiT - Virtual Try-On Studio", css=CUSTOM_CSS) as demo:
        gr.HTML(CUSTOM_CSS)
        gr.HTML(HEADER)

        with gr.Tabs() as tabs:
            # Tab 1: Quick Try-On
            with gr.TabItem("🚀 Quick Try-On", id=0):
                gr.Markdown("### Step-by-step virtual try-on experience")

                # Step indicator
                gr.HTML("""
                <div class="step-indicator">
                    <div class="step active">1️⃣ Upload</div>
                    <div class="step">→</div>
                    <div class="step">2️⃣ Adjust</div>
                    <div class="step">→</div>
                    <div class="step">3️⃣ Generate</div>
                </div>
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📸 Model Photo")
                        vton_img = gr.Image(
                            label="Upload your model image",
                            sources=['upload', 'clipboard'],
                            type="filepath",
                            height=400,
                            show_label=False
                        )
                        gr.Markdown('<p class="help-text">💡 Best: Full body, front-facing, good lighting</p>')

                    with gr.Column(scale=1):
                        gr.Markdown("#### 👕 Garment")
                        garm_img = gr.Image(
                            label="Upload garment image",
                            sources=['upload', 'clipboard'],
                            type="filepath",
                            height=400,
                            show_label=False
                        )
                        gr.Markdown('<p class="help-text">💡 Best: Clean background, clear view of garment</p>')

                with gr.Row():
                    category = gr.Dropdown(
                        label="🏷️ Garment Category",
                        choices=["Upper-body", "Lower-body", "Dresses"],
                        value="Upper-body",
                        info="Select the type of garment you want to try on"
                    )
                    resolution = gr.Dropdown(
                        label="📐 Resolution",
                        choices=["768x1024 (Fast)", "1152x1536 (Balanced)", "1536x2048 (Best)"],
                        value="1152x1536 (Balanced)",
                        info="Higher resolution = better quality but slower"
                    )

                # Quality Presets
                gr.Markdown("#### ⚡ Quick Presets")
                with gr.Row():
                    preset_fast = gr.Button("⚡ Fast (15s)", variant="secondary", size="sm")
                    preset_balanced = gr.Button("⚖️ Balanced (30s)", variant="primary", size="sm")
                    preset_quality = gr.Button("💎 Best Quality (60s)", variant="secondary", size="sm")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 🎨 Mask Adjustment")
                        masked_vton_img = gr.ImageEditor(
                            label="Draw or adjust the mask area",
                            type="numpy",
                            height=400,
                            interactive=True,
                            show_label=False,
                            brush=gr.Brush(
                                default_color="rgb(127, 127, 127)",
                                colors=["rgb(128, 128, 128)"]
                            )
                        )
                        pose_image = gr.Image(label="pose_image", visible=False, interactive=False)
                        gr.Markdown('<p class="help-text">✏️ Fine-tune the mask by drawing directly or use sliders below</p>')

                    with gr.Column(scale=1):
                        gr.Markdown("#### ✨ Results")
                        result_gallery = gr.Gallery(
                            label="Your try-on results",
                            elem_id="output-img",
                            interactive=False,
                            columns=[2],
                            rows=[2],
                            object_fit="contain",
                            height=400,
                            show_label=False
                        )
                        gr.Markdown('<p class="help-text">📥 Click on images to download</p>')

                # Action buttons
                gr.Markdown("#### 🎬 Action")
                with gr.Row():
                    run_mask_button = gr.Button("1️⃣ Generate Mask", variant="primary", size="lg", scale=1)
                    run_button = gr.Button("2️⃣ Run Try-On", variant="primary", size="lg", scale=1)
                    clear_button = gr.ClearButton(
                        components=[vton_img, garm_img, masked_vton_img, result_gallery],
                        value="🗑️ Clear All",
                        size="lg",
                        scale=1
                    )

                # Examples section
                gr.Markdown("---")
                gr.Markdown("### 🖼️ Try Our Examples")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Model Examples**")
                        with gr.Tabs():
                            with gr.TabItem("Upper-body"):
                                gr.Examples(
                                    inputs=vton_img,
                                    examples_per_page=4,
                                    examples=[
                                        os.path.join(example_path, 'model/0279.jpg'),
                                        os.path.join(example_path, 'model/0303.jpg'),
                                        os.path.join(example_path, 'model/2.jpg'),
                                        os.path.join(example_path, 'model/0083.jpg'),
                                    ])
                            with gr.TabItem("Lower-body"):
                                gr.Examples(
                                    inputs=vton_img,
                                    examples_per_page=4,
                                    examples=[
                                        os.path.join(example_path, 'model/0.jpg'),
                                        os.path.join(example_path, 'model/0179.jpg'),
                                        os.path.join(example_path, 'model/0223.jpg'),
                                        os.path.join(example_path, 'model/0347.jpg'),
                                    ])
                            with gr.TabItem("Dresses"):
                                gr.Examples(
                                    inputs=vton_img,
                                    examples_per_page=4,
                                    examples=[
                                        os.path.join(example_path, 'model/4.jpg'),
                                        os.path.join(example_path, 'model/5.jpg'),
                                        os.path.join(example_path, 'model/6.jpg'),
                                        os.path.join(example_path, 'model/7.jpg'),
                                    ])

                    with gr.Column():
                        gr.Markdown("**Garment Examples**")
                        with gr.Tabs():
                            with gr.TabItem("Upper-body"):
                                gr.Examples(
                                    inputs=garm_img,
                                    examples_per_page=4,
                                    examples=[
                                        os.path.join(example_path, 'garment/12.png'),
                                        os.path.join(example_path, 'garment/0012.jpg'),
                                        os.path.join(example_path, 'garment/0047.jpg'),
                                        os.path.join(example_path, 'garment/0049.jpg'),
                                    ])
                            with gr.TabItem("Lower-body"):
                                gr.Examples(
                                    inputs=garm_img,
                                    examples_per_page=4,
                                    examples=[
                                        os.path.join(example_path, 'garment/0317.jpg'),
                                        os.path.join(example_path, 'garment/0327.jpg'),
                                        os.path.join(example_path, 'garment/0329.jpg'),
                                        os.path.join(example_path, 'garment/0362.jpg'),
                                    ])
                            with gr.TabItem("Dresses"):
                                gr.Examples(
                                    inputs=garm_img,
                                    examples_per_page=4,
                                    examples=[
                                        os.path.join(example_path, 'garment/8.jpg'),
                                        os.path.join(example_path, 'garment/9.png'),
                                        os.path.join(example_path, 'garment/10.jpg'),
                                        os.path.join(example_path, 'garment/11.jpg'),
                                    ])

            # Tab 2: Advanced Settings
            with gr.TabItem("⚙️ Advanced Settings", id=1):
                gr.Markdown("### Fine-tune generation parameters for optimal results")

                gr.Markdown("#### 🎯 Mask Offset Adjustments")
                gr.Markdown("Use these sliders to precisely adjust the mask boundaries")
                with gr.Row():
                    offset_top = gr.Slider(
                        label="⬆️ Top Offset",
                        minimum=-200, maximum=200, step=1, value=0,
                        info="Adjust mask boundary upward/downward"
                    )
                    offset_bottom = gr.Slider(
                        label="⬇️ Bottom Offset",
                        minimum=-200, maximum=200, step=1, value=0,
                        info="Adjust mask boundary upward/downward"
                    )
                with gr.Row():
                    offset_left = gr.Slider(
                        label="⬅️ Left Offset",
                        minimum=-200, maximum=200, step=1, value=0,
                        info="Adjust mask boundary left/right"
                    )
                    offset_right = gr.Slider(
                        label="➡️ Right Offset",
                        minimum=-200, maximum=200, step=1, value=0,
                        info="Adjust mask boundary left/right"
                    )

                gr.Markdown("---")
                gr.Markdown("#### 🎨 Generation Parameters")
                with gr.Row():
                    n_steps = gr.Slider(
                        label="🔄 Inference Steps",
                        minimum=15, maximum=30, value=20, step=1,
                        info="More steps = higher quality but slower (recommended: 20-25)"
                    )
                    image_scale = gr.Slider(
                        label="🎚️ Guidance Scale",
                        minimum=1.0, maximum=5.0, value=2.5, step=0.1,
                        info="How closely to follow the input (recommended: 2.0-3.0)"
                    )

                with gr.Row():
                    seed = gr.Slider(
                        label="🎲 Random Seed",
                        minimum=-1, maximum=2147483647, step=1, value=-1,
                        info="-1 for random, or set a number for reproducible results"
                    )
                    num_images_per_prompt = gr.Slider(
                        label="🖼️ Number of Images",
                        minimum=1, maximum=4, step=1, value=1,
                        info="Generate multiple variations (more = slower)"
                    )

                with gr.Row():
                    reset_button = gr.Button("♻️ Reset to Defaults", variant="secondary")

                # Reset function
                def reset_advanced():
                    return 0, 0, 0, 0, 20, 2.5, -1, 1

                reset_button.click(
                    fn=reset_advanced,
                    outputs=[offset_top, offset_bottom, offset_left, offset_right,
                            n_steps, image_scale, seed, num_images_per_prompt]
                )

            # Tab 3: User Guide
            with gr.TabItem("📖 User Guide", id=2):
                gr.HTML(GUIDE_CONTENT)

        # Parse resolution string to extract actual values
        def parse_resolution(res_str):
            if "768x1024" in res_str:
                return "768x1024"
            elif "1152x1536" in res_str:
                return "1152x1536"
            elif "1536x2048" in res_str:
                return "1536x2048"
            return "1152x1536"

        # Wrapper functions to handle resolution parsing
        def generate_mask_wrapper(vton_img, category, offset_top, offset_bottom, offset_left, offset_right):
            return generator.generate_mask(vton_img, category, offset_top, offset_bottom, offset_left, offset_right)

        def process_wrapper(vton_img, garm_img, pre_mask, pose_image, n_steps, image_scale, seed, num_images, resolution):
            actual_resolution = parse_resolution(resolution)
            return generator.process(vton_img, garm_img, pre_mask, pose_image, n_steps, image_scale, seed, num_images, actual_resolution)

        # Preset button handlers
        def apply_fast_preset():
            return 15, 2.0, "768x1024 (Fast)"

        def apply_balanced_preset():
            return 20, 2.5, "1152x1536 (Balanced)"

        def apply_quality_preset():
            return 30, 3.0, "1536x2048 (Best)"

        preset_fast.click(fn=apply_fast_preset, outputs=[n_steps, image_scale, resolution])
        preset_balanced.click(fn=apply_balanced_preset, outputs=[n_steps, image_scale, resolution])
        preset_quality.click(fn=apply_quality_preset, outputs=[n_steps, image_scale, resolution])

        # Main functionality
        ips1 = [vton_img, category, offset_top, offset_bottom, offset_left, offset_right]
        ips2 = [vton_img, garm_img, masked_vton_img, pose_image, n_steps, image_scale, seed, num_images_per_prompt, resolution]

        run_mask_button.click(fn=generate_mask_wrapper, inputs=ips1, outputs=[masked_vton_img, pose_image])
        run_button.click(fn=process_wrapper, inputs=ips2, outputs=[result_gallery])

    return demo

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FitDiT")
    parser.add_argument("--model_path", type=str, required=True, help="The path of FitDiT model.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--fp16", action="store_true", help="Load model with fp16, default is bf16")
    parser.add_argument("--offload", action="store_true", help="Offload model to CPU when not in use.")
    parser.add_argument("--aggressive_offload", action="store_true", help="Offload model more aggressively to CPU when not in use.")
    args = parser.parse_args()
    demo = create_demo(args.model_path, args.device, args.offload, args.aggressive_offload, args.fp16)
    demo.launch(share=True)
