"""
Edit Module

Provides image editing workflows with 1, 2, or 3 reference images.
Supports multiple edit models (Flux2 Klein 4B/9B, Z-Image Edit, etc.)
Uses the ReferenceLatent system for powerful image-guided editing.
"""

import json
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import gradio as gr
import httpx

from modules.lora_ui import (
    create_lora_ui,
    setup_lora_handlers,
    get_lora_inputs,
    get_lora_params,
    ensure_dummy_lora,
)
from modules.model_ui import (
    create_model_ui,
    create_quick_preset_selector,
    setup_model_handlers,
    get_model_inputs,
    BASE_MODEL_TYPES,
)

if TYPE_CHECKING:
    from modules import SharedServices

logger = logging.getLogger(__name__)

# Module metadata
TAB_ID = "edit"
TAB_LABEL = "✏️ Edit"
TAB_ORDER = 1  # After Z-Image, before Upscale

# Default samplers (fallback if ComfyUI not available)
DEFAULT_SAMPLERS = ["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_sde"]

# Session temp directory for results
_results_temp_dir = tempfile.TemporaryDirectory(prefix="edit_results_")


def new_random_seed():
    """Generate a new random seed."""
    return random.randint(0, 999999999999)


def fetch_comfyui_samplers(kit) -> list:
    """Fetch available samplers from ComfyUI."""
    if kit is None:
        return DEFAULT_SAMPLERS.copy()
    try:
        with httpx.Client(timeout=5) as client:
            response = client.get(f"{kit.comfyui_url}/object_info/KSampler")
            if response.status_code == 200:
                data = response.json()
                ksampler = data.get("KSampler", {}).get("input", {}).get("required", {})
                sampler_info = ksampler.get("sampler_name", [])
                if sampler_info and isinstance(sampler_info[0], list):
                    return sampler_info[0]
    except Exception as e:
        logger.warning(f"Could not fetch samplers: {e}")
    return DEFAULT_SAMPLERS.copy()


async def download_image_from_url(url: str) -> str:
    """Download image from ComfyUI URL to a local temp file."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        suffix = Path(url).suffix or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(response.content)
            return f.name


def copy_to_temp_with_name(image_path: str, prompt: str, seed: int) -> str:
    """Copy image to session temp dir with a meaningful name."""
    timestamp = datetime.now().strftime("%H%M%S")
    safe_prompt = "".join(c if c.isalnum() or c in " -_" else "" for c in prompt[:30]).strip()
    safe_prompt = safe_prompt.replace(" ", "_") if safe_prompt else "edit"
    filename = f"{safe_prompt}_{seed}_{timestamp}.png"
    temp_path = Path(_results_temp_dir.name) / filename
    shutil.copy2(image_path, temp_path)
    return str(temp_path)


def save_to_outputs(image_path: str, prompt: str, outputs_dir: Path) -> str:
    """Save image to outputs/edit folder."""
    timestamp = datetime.now().strftime("%H%M%S")
    safe_prompt = "".join(c if c.isalnum() or c in " -_" else "" for c in prompt[:30]).strip()
    safe_prompt = safe_prompt.replace(" ", "_") if safe_prompt else "edit"
    target_dir = outputs_dir / "edit"
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{safe_prompt}_{timestamp}.png"
    output_path = target_dir / filename
    shutil.copy2(image_path, output_path)
    logger.info(f"Saved to: {output_path}")
    return str(output_path)


def open_folder(folder_path: Path):
    """Cross-platform folder opener."""
    folder_path.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        os.startfile(folder_path)
    elif sys.platform == "darwin":
        subprocess.run(["open", str(folder_path)])
    else:
        subprocess.run(["xdg-open", str(folder_path)])


def get_workflow_file(num_inputs: int, use_gguf: bool) -> str:
    """Get workflow filename based on number of input images and GGUF mode."""
    suffix = "_gguf" if use_gguf else ""
    return f"klein_edit_{num_inputs}_input{suffix}.json"


# =============================================================================
# Prompt Library
# =============================================================================

PROMPT_LIBRARY_FILE = "edit_prompts.json"

DEFAULT_PROMPTS = {
    "Style Transfer": "Transform the image to match the artistic style while preserving the subject",
    "Realistic Enhancement": "Make the image more photorealistic with natural lighting and textures",
    "Anime/Illustration": "Convert to anime or illustrated style while keeping the composition",
    "Oil Painting": "Transform into an oil painting with visible brushstrokes and rich colors",
    "Watercolor": "Convert to a soft watercolor painting style",
    "Sketch": "Transform into a pencil sketch or line drawing",
    "Cinematic": "Apply cinematic color grading and dramatic lighting",
    "Vintage/Retro": "Apply a vintage or retro aesthetic with muted colors",
    "Fantasy": "Transform into a fantasy art style with magical elements",
    "Cyberpunk": "Apply cyberpunk aesthetic with neon colors and futuristic elements",
}


def get_prompt_library_path(app_dir: Path) -> Path:
    return app_dir / "modules" / PROMPT_LIBRARY_FILE


def load_prompt_library(app_dir: Path) -> dict:
    path = get_prompt_library_path(app_dir)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load prompt library: {e}")
    return DEFAULT_PROMPTS.copy()


def save_prompt_library(app_dir: Path, prompts: dict) -> bool:
    path = get_prompt_library_path(app_dir)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(prompts, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Failed to save prompt library: {e}")
        return False


def get_prompt_choices(prompts: dict) -> list:
    return [""] + sorted(prompts.keys())


# =============================================================================
# Workflow Execution
# =============================================================================

async def run_edit_workflow(
    services: "SharedServices",
    num_inputs: int,
    image1: str,
    image2: str,
    image3: str,
    prompt: str,
    negative_prompt: str,
    seed: int,
    randomize_seed: bool,
    steps: int,
    cfg: float,
    megapixels: float,
    sampler_name: str,
    # Model inputs from model_ui
    clip_type: str,
    use_gguf: bool,
    unet_name: str,
    clip_name: str,
    vae_name: str,
    autosave: bool,
    # LoRA params
    lora1_enabled: bool, lora1_name: str, lora1_strength: float,
    lora2_enabled: bool, lora2_name: str, lora2_strength: float,
    lora3_enabled: bool, lora3_name: str, lora3_strength: float,
    lora4_enabled: bool, lora4_name: str, lora4_strength: float,
    lora5_enabled: bool, lora5_name: str, lora5_strength: float,
    lora6_enabled: bool, lora6_name: str, lora6_strength: float,
):
    """Execute edit workflow. Yields (slider_tuple, status, seed, result_path)."""
    actual_seed = new_random_seed() if randomize_seed else int(seed)
    outputs_dir = services.get_outputs_dir()
    
    # Validate required images
    if image1 is None:
        yield None, "❌ Please upload Image 1 (primary input)", actual_seed, None
        return
    if num_inputs >= 2 and image2 is None:
        yield None, "❌ Please upload Image 2", actual_seed, None
        return
    if num_inputs >= 3 and image3 is None:
        yield None, "❌ Please upload Image 3", actual_seed, None
        return
    
    if not prompt or not prompt.strip():
        yield None, "❌ Please enter an edit instruction", actual_seed, None
        return
    
    # Validate models
    if not unet_name:
        yield None, "❌ Please select a diffusion model", actual_seed, None
        return
    if not clip_name:
        yield None, "❌ Please select a text encoder", actual_seed, None
        return
    
    yield None, "⏳ Editing...", actual_seed, None
    
    workflow_file = get_workflow_file(num_inputs, use_gguf)
    workflow_path = services.workflows_dir / workflow_file
    
    if not workflow_path.exists():
        yield None, f"❌ Workflow not found: {workflow_file}", actual_seed, None
        return
    
    lora_params = get_lora_params(
        lora1_enabled, lora1_name, lora1_strength,
        lora2_enabled, lora2_name, lora2_strength,
        lora3_enabled, lora3_name, lora3_strength,
        lora4_enabled, lora4_name, lora4_strength,
        lora5_enabled, lora5_name, lora5_strength,
        lora6_enabled, lora6_name, lora6_strength,
    )
    
    params = {
        "image1": image1,
        "prompt": prompt.strip(),
        "negative_prompt": negative_prompt.strip() if negative_prompt else "",
        "seed": int(actual_seed),
        "steps": int(steps),
        "cfg": float(cfg),
        "megapixels": float(megapixels),
        "sampler_name": sampler_name,
        "unet_name": unet_name,
        "clip_name": clip_name,
        "vae_name": vae_name,
    }
    
    if num_inputs >= 2:
        params["image2"] = image2
    if num_inputs >= 3:
        params["image3"] = image3
    
    params.update(lora_params)
    
    try:
        result = await services.kit.execute(str(workflow_path), params)
        
        if result.status == "error":
            yield None, f"❌ {result.msg}", actual_seed, None
            return
        
        if not result.images:
            yield None, "❌ No output generated", actual_seed, None
            return
        
        image_path = result.images[0]
        if image_path.startswith("http"):
            image_path = await download_image_from_url(image_path)
        
        image_path = copy_to_temp_with_name(image_path, prompt, actual_seed)
        
        if autosave:
            save_to_outputs(image_path, prompt, outputs_dir)
        
        status = f"✓ {result.duration:.1f}s" if result.duration else "✓ Done"
        if autosave:
            status += " | Saved"
        
        yield (image1, image_path), status, actual_seed, image_path
        
    except Exception as e:
        logger.error(f"Edit error: {e}", exc_info=True)
        yield None, f"❌ {str(e)}", actual_seed, None



# =============================================================================
# Tab Creation
# =============================================================================

def create_tab(services: "SharedServices") -> gr.TabItem:
    """Create the Edit tab with full model selection."""
    
    loras_dir = services.models_dir / "loras"
    outputs_dir = services.get_outputs_dir()
    edit_outputs_dir = outputs_dir / "edit"
    
    ensure_dummy_lora(loras_dir)
    samplers = fetch_comfyui_samplers(services.kit)
    prompt_library = load_prompt_library(services.app_dir)
    
    with gr.TabItem(TAB_LABEL, id=TAB_ID) as tab:
        gr.Markdown("*Edit images using reference latents. Use 1, 2, or 3 input images.*")
        
        with gr.Row():
            # ===== LEFT COLUMN =====
            with gr.Column(scale=1):
                with gr.Tabs() as edit_tabs:
                    # --- 1-Input Tab ---
                    with gr.TabItem("📷 1 Image", id="edit_1"):
                        image1_1 = gr.Image(label="Input Image", type="filepath", elem_classes="image-window")
                        prompt_1 = gr.Textbox(
                            label="Edit Instruction",
                            placeholder="e.g., change to realistic style, make it look like a painting...",
                            lines=2
                        )
                        prompt_select_1 = gr.Dropdown(
                            label="Load Saved Prompt",
                            choices=get_prompt_choices(prompt_library),
                            value="",
                            allow_custom_value=False
                        )
                        with gr.Row():
                            generate_1_btn = gr.Button("✏️ Edit", variant="primary", scale=3)
                            stop_1_btn = gr.Button("⏹️ Stop", size="sm", variant="stop", scale=1)
                    
                    # --- 2-Input Tab ---
                    with gr.TabItem("📷📷 2 Images", id="edit_2"):
                        with gr.Row():
                            image1_2 = gr.Image(label="Image 1 (Primary)", type="filepath", height=280)
                            image2_2 = gr.Image(label="Image 2 (Reference)", type="filepath", height=280)
                        prompt_2 = gr.Textbox(
                            label="Edit Instruction",
                            placeholder="e.g., Change image 1 to match the style of image 2...",
                            lines=2
                        )
                        prompt_select_2 = gr.Dropdown(
                            label="Load Saved Prompt",
                            choices=get_prompt_choices(prompt_library),
                            value="",
                            allow_custom_value=False
                        )
                        with gr.Row():
                            generate_2_btn = gr.Button("✏️ Edit", variant="primary", scale=3)
                            stop_2_btn = gr.Button("⏹️ Stop", size="sm", variant="stop", scale=1)
                    
                    # --- 3-Input Tab ---
                    with gr.TabItem("📷📷📷 3 Images", id="edit_3"):
                        with gr.Row():
                            image1_3 = gr.Image(label="Image 1 (Primary)", type="filepath", height=240)
                            image2_3 = gr.Image(label="Image 2 (Ref A)", type="filepath", height=240)
                            image3_3 = gr.Image(label="Image 3 (Ref B)", type="filepath", height=240)
                        prompt_3 = gr.Textbox(
                            label="Edit Instruction",
                            placeholder="e.g., Combine style from image 2 with background from image 3...",
                            lines=2
                        )
                        prompt_select_3 = gr.Dropdown(
                            label="Load Saved Prompt",
                            choices=get_prompt_choices(prompt_library),
                            value="",
                            allow_custom_value=False
                        )
                        with gr.Row():
                            generate_3_btn = gr.Button("✏️ Edit", variant="primary", scale=3)
                            stop_3_btn = gr.Button("⏹️ Stop", size="sm", variant="stop", scale=1)
                
                # === SHARED SETTINGS ===
                with gr.Accordion("⚙️ Settings", open=False):
                    with gr.Row():
                        steps = gr.Slider(label="Steps", value=4, minimum=1, maximum=20, step=1)
                        cfg = gr.Slider(label="CFG", value=1.0, minimum=1.0, maximum=5.0, step=0.1)
                    with gr.Row():
                        megapixels = gr.Slider(label="Megapixels", value=1.0, minimum=0.5, maximum=2.0, step=0.1)
                        sampler = gr.Dropdown(label="Sampler", choices=samplers, value="euler")
                    with gr.Row():
                        seed = gr.Number(label="Seed", value=new_random_seed(), minimum=0, step=1, scale=2)
                        randomize_seed = gr.Checkbox(label="🎲", value=True, scale=0, min_width=50)
                    negative = gr.Textbox(label="Negative Prompt", value="", lines=1)
                
                # Quick model preset selector (outside accordion for easy access)
                # edit_only=True filters to only show presets that support edit workflows (Klein models)
                # default_to_manual=True starts with "Manual" mode using current model selections
                quick_preset, clip_type_state, presets_state = create_quick_preset_selector(
                    settings_manager=services.settings,
                    label="Model Preset",
                    edit_only=True,
                    default_to_manual=True,
                )
                
                # === MODEL SELECTION (full model_ui) ===
                model_components = create_model_ui(
                    models_dir=services.models_dir,
                    accordion_label="🔧 Models",
                    accordion_open=False,
                    settings_manager=services.settings,
                    quick_preset_dropdown=quick_preset,
                    clip_type_state=clip_type_state,
                    presets_state=presets_state,
                    edit_only=True,  # Only show edit-compatible base types (Klein models)
                )
                
                lora_components = create_lora_ui(loras_dir, accordion_open=False)
            
            # ===== RIGHT COLUMN =====
            with gr.Column(scale=1):
                output_slider = gr.ImageSlider(
                    label="Before / After",
                    type="filepath",
                    elem_classes="image-window",
                    show_download_button=True
                )
                with gr.Row():
                    save_btn = gr.Button("💾 Save", size="sm", variant="primary")
                    send_btn = gr.Button("🔍 Send to SeedVR2", size="sm", variant="huggingface")
                with gr.Row():
                    autosave = gr.Checkbox(label="Auto-save", container=False, value=False)
                    open_folder_btn = gr.Button("📂 Open Folder", size="sm")
                
                status = gr.Textbox(label="Status", interactive=False, show_label=False, lines=1)
                
                from modules.system_monitor_ui import create_monitor_textboxes
                gpu_monitor, cpu_monitor = create_monitor_textboxes()
                
                result_path = gr.State(value=None)
                
                # === PROMPT LIBRARY ===
                with gr.Accordion("📝 Prompt Library", open=False):
                    gr.Markdown("""
*Save and reuse your favorite edit prompts. Tips for effective prompts:*
- Be specific about the desired style or transformation
- Mention what to preserve (e.g., "keep the subject's pose")
- For multi-image edits, reference which image provides what
""")
                    with gr.Row():
                        new_prompt_name = gr.Textbox(label="Prompt Name", placeholder="e.g., Cinematic Portrait", scale=1)
                        new_prompt_text = gr.Textbox(label="Prompt Text", placeholder="Enter the prompt to save...", scale=2)
                    with gr.Row():
                        save_prompt_btn = gr.Button("💾 Save Prompt", size="sm", variant="primary")
                        delete_prompt_btn = gr.Button("🗑️ Delete Selected", size="sm", variant="stop")
                    prompt_library_status = gr.Textbox(label="", show_label=False, interactive=False, lines=1)
        
        # ===== EVENT HANDLERS =====
        
        # Setup model handlers (edit_only=True to filter presets/base types)
        setup_model_handlers(model_components, services.models_dir, services.settings, edit_only=True)
        
        # Setup LoRA handlers
        setup_lora_handlers(lora_components, loras_dir)
        lora_inputs = get_lora_inputs(lora_components)
        
        # Get model inputs from model_ui
        model_inputs = get_model_inputs(model_components)
        
        # Shared settings inputs
        shared_inputs = [negative, seed, randomize_seed, steps, cfg, megapixels, sampler] + model_inputs + [autosave] + lora_inputs
        
        async def stop_generation():
            try:
                async with httpx.AsyncClient() as client:
                    await client.post(f"{services.kit.comfyui_url}/interrupt")
                return "⏹️ Stopping..."
            except Exception as e:
                return f"⏹️ Stop requested ({e})"
        
        def save_result(res_path, p1, p2, p3):
            if not res_path:
                return "❌ No image to save"
            prompt = p1 or p2 or p3 or "edit"
            save_to_outputs(res_path, prompt, outputs_dir)
            return "✓ Saved"
        
        # Prompt library handlers
        def on_prompt_select(name):
            if not name:
                return ""
            library = load_prompt_library(services.app_dir)
            return library.get(name, "")
        
        def on_save_prompt(name, text):
            if not name or not name.strip():
                return "❌ Enter a prompt name", gr.update(), gr.update(), gr.update()
            if not text or not text.strip():
                return "❌ Enter prompt text", gr.update(), gr.update(), gr.update()
            
            library = load_prompt_library(services.app_dir)
            library[name.strip()] = text.strip()
            
            if save_prompt_library(services.app_dir, library):
                choices = get_prompt_choices(library)
                return f"✓ Saved '{name.strip()}'", gr.update(choices=choices), gr.update(choices=choices), gr.update(choices=choices)
            return "❌ Failed to save", gr.update(), gr.update(), gr.update()
        
        def on_delete_prompt(name):
            if not name:
                return "❌ Select a prompt to delete", gr.update(), gr.update(), gr.update()
            
            library = load_prompt_library(services.app_dir)
            if name in library:
                del library[name]
                save_prompt_library(services.app_dir, library)
                choices = get_prompt_choices(library)
                return f"✓ Deleted '{name}'", gr.update(choices=choices, value=""), gr.update(choices=choices, value=""), gr.update(choices=choices, value="")
            return "❌ Prompt not found", gr.update(), gr.update(), gr.update()
        
        # Wire prompt selection
        prompt_select_1.change(fn=on_prompt_select, inputs=[prompt_select_1], outputs=[prompt_1])
        prompt_select_2.change(fn=on_prompt_select, inputs=[prompt_select_2], outputs=[prompt_2])
        prompt_select_3.change(fn=on_prompt_select, inputs=[prompt_select_3], outputs=[prompt_3])
        
        save_prompt_btn.click(
            fn=on_save_prompt,
            inputs=[new_prompt_name, new_prompt_text],
            outputs=[prompt_library_status, prompt_select_1, prompt_select_2, prompt_select_3]
        )
        delete_prompt_btn.click(
            fn=on_delete_prompt,
            inputs=[prompt_select_1],
            outputs=[prompt_library_status, prompt_select_1, prompt_select_2, prompt_select_3]
        )
        
        # Edit handlers
        async def run_edit_1(img1, prompt, neg, seed_val, rand, steps_val, cfg_val, mp, samp,
                             clip_type, use_gguf, unet, clip, vae, auto, *lora_args):
            async for result in run_edit_workflow(
                services, 1, img1, None, None, prompt, neg, seed_val, rand,
                steps_val, cfg_val, mp, samp, clip_type, use_gguf, unet, clip, vae, auto, *lora_args
            ):
                yield result
        
        async def run_edit_2(img1, img2, prompt, neg, seed_val, rand, steps_val, cfg_val, mp, samp,
                             clip_type, use_gguf, unet, clip, vae, auto, *lora_args):
            async for result in run_edit_workflow(
                services, 2, img1, img2, None, prompt, neg, seed_val, rand,
                steps_val, cfg_val, mp, samp, clip_type, use_gguf, unet, clip, vae, auto, *lora_args
            ):
                yield result
        
        async def run_edit_3(img1, img2, img3, prompt, neg, seed_val, rand, steps_val, cfg_val, mp, samp,
                             clip_type, use_gguf, unet, clip, vae, auto, *lora_args):
            async for result in run_edit_workflow(
                services, 3, img1, img2, img3, prompt, neg, seed_val, rand,
                steps_val, cfg_val, mp, samp, clip_type, use_gguf, unet, clip, vae, auto, *lora_args
            ):
                yield result
        
        # Wire edit buttons
        generate_1_btn.click(fn=run_edit_1, inputs=[image1_1, prompt_1] + shared_inputs, outputs=[output_slider, status, seed, result_path])
        stop_1_btn.click(fn=stop_generation, outputs=[status])
        
        generate_2_btn.click(fn=run_edit_2, inputs=[image1_2, image2_2, prompt_2] + shared_inputs, outputs=[output_slider, status, seed, result_path])
        stop_2_btn.click(fn=stop_generation, outputs=[status])
        
        generate_3_btn.click(fn=run_edit_3, inputs=[image1_3, image2_3, image3_3, prompt_3] + shared_inputs, outputs=[output_slider, status, seed, result_path])
        stop_3_btn.click(fn=stop_generation, outputs=[status])
        
        save_btn.click(fn=save_result, inputs=[result_path, prompt_1, prompt_2, prompt_3], outputs=[status])
        open_folder_btn.click(fn=lambda: open_folder(edit_outputs_dir))
        
        # Register components
        services.inter_module.register_component("edit_send_btn", send_btn)
        services.inter_module.register_component("edit_result_path", result_path)
        services.inter_module.register_component("edit_status", status)
        services.inter_module.register_component("edit_gpu_monitor", gpu_monitor)
        services.inter_module.register_component("edit_cpu_monitor", cpu_monitor)
        
        services.inter_module.image_transfer.register_receiver(
            tab_id=TAB_ID,
            label=TAB_LABEL,
            input_component=image1_1,
            status_component=status
        )
    
    return tab
