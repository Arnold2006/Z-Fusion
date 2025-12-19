"""
Experimental Module

Provides a sandbox environment for testing new ComfyUI workflows before
production integration. First workflow: z_image_upscaleAny.json using
the FlowMatchEulerDiscreteScheduler custom node.

This module manages its own custom node dependencies, avoiding bloat
in the main install/update scripts.
"""

import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import gradio as gr
import httpx

if TYPE_CHECKING:
    from modules import SharedServices

logger = logging.getLogger(__name__)

# Module metadata
TAB_ID = "experimental"
TAB_LABEL = "🧪 Experimental"
TAB_ORDER = 2

# Custom node configuration
CUSTOM_NODE_NAME = "ComfyUI-EulerDiscreteScheduler"
CUSTOM_NODE_REPO = "https://github.com/erosDiffusion/ComfyUI-EulerDiscreteScheduler.git"

# Workflow configuration
UPSCALE_WORKFLOW = "z_image_upscaleAny.json"

# Status message constants
STATUS_UPSCALING = "⏳ Upscaling..."
STATUS_SUCCESS_PREFIX = "✓"
STATUS_ERROR_PREFIX = "❌"

# Default samplers (fallback if ComfyUI not available)
DEFAULT_SAMPLERS = ["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_sde"]

# Model defaults - Standard
DEFAULT_DIFFUSION = "z_image_turbo_bf16.safetensors"
DEFAULT_CLIP = "qwen_3_4b.safetensors"
DEFAULT_VAE = "ae.safetensors"

# Model defaults - GGUF
DEFAULT_DIFFUSION_GGUF = "z-image-turbo-q4_k_m.gguf"
DEFAULT_CLIP_GGUF = "Qwen3-4B-Q4_K_M.gguf"

# File extensions by mode
STANDARD_EXTENSIONS = (".safetensors", ".ckpt", ".pt")
GGUF_EXTENSIONS = (".gguf",)
MODEL_EXTENSIONS = (".safetensors", ".ckpt", ".pt", ".gguf")

# Name filters for Z-Image compatible models
ZIMAGE_FILTERS = {
    "diffusion": "z",  # z-image, z_image variants
    "text_encoder": "qwen",  # Qwen3 text encoder
    "vae": "ae",  # ae.safetensors
}


def scan_models(folder: Path, extensions: tuple = MODEL_EXTENSIONS, name_filter: str = None) -> list:
    """Scan folder recursively for model files, returning relative paths."""
    if not folder.exists():
        return []
    models = []
    for ext in extensions:
        for f in folder.rglob(f"*{ext}"):
            rel_path = str(f.relative_to(folder))
            if name_filter is None or name_filter.lower() in rel_path.lower():
                models.append(rel_path)
    return sorted(models)


def get_default_model(choices: list, preferred: str) -> str:
    """Get default model, preferring the specified one if available."""
    if preferred in choices:
        return preferred
    return choices[0] if choices else preferred


def get_models_by_mode(folder: Path, is_gguf: bool, default_standard: str, default_gguf: str, name_filter: str = None) -> list:
    """Get models filtered by mode (standard vs GGUF) and optional name filter."""
    extensions = GGUF_EXTENSIONS if is_gguf else STANDARD_EXTENSIONS
    default = default_gguf if is_gguf else default_standard
    models = scan_models(folder, extensions, name_filter)
    return models or [default]


def get_upscale_workflow(use_gguf: bool) -> str:
    """Get the appropriate upscale workflow based on GGUF mode."""
    if use_gguf:
        return "z_image_upscaleAny_gguf.json"
    return "z_image_upscaleAny.json"


def format_status_success(duration: float, saved: bool = False) -> str:
    """Format status message for successful upscale operation."""
    if saved:
        return f"{STATUS_SUCCESS_PREFIX} {duration:.1f}s | Saved"
    return f"{STATUS_SUCCESS_PREFIX} {duration:.1f}s"


def format_status_error(error_message: str) -> str:
    """Format status message for failed upscale operation."""
    return f"{STATUS_ERROR_PREFIX} {error_message}"


def new_random_seed():
    """Generate a new random seed."""
    return random.randint(0, 999999999999)


def check_custom_node_installed(custom_nodes_dir: Path, node_name: str) -> bool:
    """Check if a custom node directory exists."""
    node_path = custom_nodes_dir / node_name
    return node_path.exists() and node_path.is_dir()


def install_custom_node(custom_nodes_dir: Path, repo_url: str, node_name: str) -> tuple[bool, str]:
    """Clone a custom node repository."""
    node_path = custom_nodes_dir / node_name
    
    if node_path.exists():
        return False, f"Custom node '{node_name}' already exists"
    
    try:
        custom_nodes_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Cloning {repo_url} to {node_path}")
        result = subprocess.run(
            ["git", "clone", repo_url, str(node_path)],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip() or "Unknown error"
            logger.error(f"Git clone failed: {error_msg}")
            return False, f"Failed to clone: {error_msg}"
        
        logger.info(f"Successfully installed {node_name}")
        return True, f"✓ Installed '{node_name}'. Please restart ComfyUI to load the node."
        
    except subprocess.TimeoutExpired:
        return False, "Installation timed out. Check your network connection."
    except FileNotFoundError:
        return False, "Git is not installed or not in PATH"
    except Exception as e:
        return False, f"Installation failed: {str(e)}"


def update_custom_node(custom_nodes_dir: Path, node_name: str) -> tuple[bool, str]:
    """Git pull an existing custom node."""
    node_path = custom_nodes_dir / node_name
    
    if not node_path.exists():
        return False, f"Custom node '{node_name}' is not installed"
    
    try:
        logger.info(f"Updating {node_name} via git pull")
        result = subprocess.run(
            ["git", "pull"],
            cwd=str(node_path),
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip() or "Unknown error"
            return False, f"Failed to update: {error_msg}"
        
        output = result.stdout.strip()
        if "Already up to date" in output:
            return True, f"✓ '{node_name}' is already up to date"
        
        return True, f"✓ Updated '{node_name}'. Please restart ComfyUI to apply changes."
        
    except subprocess.TimeoutExpired:
        return False, "Update timed out. Check your network connection."
    except FileNotFoundError:
        return False, "Git is not installed or not in PATH"
    except Exception as e:
        return False, f"Update failed: {str(e)}"


async def download_image_from_url(url: str) -> str:
    """Download image from ComfyUI URL to a local temp file."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        suffix = Path(url).suffix or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(response.content)
            return f.name


def save_experimental_output(image_path: str, original_path: str, outputs_dir: Path) -> str:
    """Save upscaled image to outputs/experimental folder."""
    timestamp = datetime.now().strftime("%H%M%S")
    
    if original_path:
        original_stem = Path(original_path).stem[:30]
        safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in original_stem)
    else:
        safe_stem = "image"
    
    target_dir = outputs_dir / "experimental"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{safe_stem}_upscaled_{timestamp}.png"
    output_path = target_dir / filename
    
    shutil.copy2(image_path, output_path)
    logger.info(f"Saved experimental output to: {output_path}")
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


async def experimental_upscale(
    services: "SharedServices",
    input_image: str,
    prompt: str,
    seed: int,
    randomize_seed: bool,
    megapixels: float,
    scale_by: float,
    steps: int,
    start_at_step: int,
    end_at_step: int,
    shift: float,
    cfg: float,
    sampler_name: str,
    # Model params
    use_gguf: bool,
    unet_name: str,
    clip_name: str,
    vae_name: str,
    # Advanced scheduler params
    base_shift: float,
    max_shift: float,
    use_karras_sigmas: str,
    stochastic_sampling: str,
    autosave: bool,
):
    """
    Execute the z_image_upscaleAny workflow.
    Yields (slider_tuple, status_message, actual_seed, result_path) tuples.
    """
    outputs_dir = services.get_outputs_dir()
    start_time = time.time()
    
    # Handle seed
    actual_seed = new_random_seed() if randomize_seed else int(seed)
    
    # Input validation
    if input_image is None:
        yield None, format_status_error("Please upload an image to upscale"), actual_seed, None
        return
    
    # Select workflow based on GGUF mode
    workflow_file = get_upscale_workflow(use_gguf)
    workflow_path = services.workflows_dir / workflow_file
    if not workflow_path.exists():
        yield None, format_status_error(f"Workflow not found: {workflow_file}"), actual_seed, None
        return
    
    logger.info(f"Experimental upscale: megapixels={megapixels}, scale_by={scale_by}, "
                f"steps={steps}, start={start_at_step}, end={end_at_step}, "
                f"shift={shift}, cfg={cfg}, seed={actual_seed}")
    
    # Yield initial status
    yield None, STATUS_UPSCALING, actual_seed, None
    
    try:
        # Build params dict
        params = {
            "image": input_image,
            "prompt": prompt.strip() if prompt else "",
            "seed": int(actual_seed),
            "cfg": float(cfg),
            "scale_by": float(scale_by),
            "megapixels": float(megapixels),
            "steps": int(steps),
            "start_at_step": int(start_at_step),
            "end_at_step": int(end_at_step),
            "shift": float(shift),
            "sampler_name": sampler_name,
            # Model params
            "unet_name": unet_name,
            "clip_name": clip_name,
            "vae_name": vae_name,
            # Advanced scheduler params
            "base_shift": float(base_shift),
            "max_shift": float(max_shift),
            "use_karras_sigmas": use_karras_sigmas,
            "stochastic_sampling": stochastic_sampling,
        }
        
        # Execute workflow
        result = await services.kit.execute(str(workflow_path), params)
        
        if result.status == "error":
            yield None, format_status_error(f"Upscale failed: {result.msg}"), actual_seed, None
            return
        
        if not result.images:
            yield None, format_status_error("No images generated"), actual_seed, None
            return
        
        # Get the result image path
        image_path = result.images[0]
        if image_path.startswith("http"):
            image_path = await download_image_from_url(image_path)
        
        duration = time.time() - start_time
        
        # Autosave if enabled
        result_path = image_path
        if autosave:
            result_path = save_experimental_output(image_path, input_image, outputs_dir)
            status = format_status_success(duration, saved=True)
        else:
            status = format_status_success(duration, saved=False)
        
        # Return tuple for ImageSlider (original, upscaled)
        yield (input_image, image_path), status, actual_seed, result_path
        
    except Exception as e:
        logger.error(f"Experimental upscale error: {e}", exc_info=True)
        if "connect" in str(e).lower():
            yield None, format_status_error("Cannot connect to ComfyUI"), actual_seed, None
        else:
            yield None, format_status_error(str(e)), actual_seed, None



def create_tab(services: "SharedServices") -> gr.TabItem:
    """Create the Experimental tab with sub-tabs for different workflows."""
    custom_nodes_dir = services.app_dir / "comfyui" / "custom_nodes"
    outputs_dir = services.get_outputs_dir()
    experimental_dir = outputs_dir / "experimental"
    
    # Model directories
    diffusion_dir = services.models_dir / "diffusion_models"
    text_encoders_dir = services.models_dir / "text_encoders"
    vae_dir = services.models_dir / "vae"
    
    # Check what models are available to determine default mode
    has_standard_diffusion = bool(scan_models(diffusion_dir, STANDARD_EXTENSIONS, ZIMAGE_FILTERS["diffusion"]))
    has_gguf_diffusion = bool(scan_models(diffusion_dir, GGUF_EXTENSIONS, ZIMAGE_FILTERS["diffusion"]))
    default_gguf_mode = has_gguf_diffusion and not has_standard_diffusion
    
    # Initial model scan (will be refreshed when mode changes)
    diffusion_models = get_models_by_mode(diffusion_dir, default_gguf_mode, DEFAULT_DIFFUSION, DEFAULT_DIFFUSION_GGUF, ZIMAGE_FILTERS["diffusion"])
    clip_models = get_models_by_mode(text_encoders_dir, default_gguf_mode, DEFAULT_CLIP, DEFAULT_CLIP_GGUF, ZIMAGE_FILTERS["text_encoder"])
    vae_models = scan_models(vae_dir, STANDARD_EXTENSIONS, ZIMAGE_FILTERS["vae"]) or [DEFAULT_VAE]
    
    # Check initial installation status
    is_installed = check_custom_node_installed(custom_nodes_dir, CUSTOM_NODE_NAME)
    
    # Fetch available samplers from ComfyUI
    samplers = DEFAULT_SAMPLERS.copy()
    try:
        with httpx.Client(timeout=5) as client:
            response = client.get(f"{services.kit.comfyui_url}/object_info/KSamplerSelect")
            if response.status_code == 200:
                data = response.json()
                sampler_info = data.get("KSamplerSelect", {}).get("input", {}).get("required", {}).get("sampler_name", [])
                if sampler_info and isinstance(sampler_info[0], list):
                    samplers = sampler_info[0]
    except Exception:
        pass
    
    with gr.TabItem(TAB_LABEL, id=TAB_ID) as tab:
        gr.Markdown("## 🧪 Experimental Workflows")
        gr.Markdown("*Sandbox for testing new workflows before production integration.*")
        
        # Sub-tabs for different experimental workflows
        with gr.Tabs():
            with gr.TabItem("🔍 UpscaleAny", id="upscale_any"):
                with gr.Row():
                    # ===== LEFT COLUMN - Input =====
                    with gr.Column(scale=1):
                        # Input image
                        input_image = gr.Image(
                            label="Input Image",
                            type="filepath",
                            elem_classes="image-window"
                        )
                        
                        # Prompt textbox
                        prompt = gr.Textbox(
                            label="Prompt (Optional)",
                            placeholder="Leave empty, or feel free to experiment...",
                            lines=2,
                            info="Guide the upscale with a description"
                        )
                        
                        # Upscale button
                        upscale_btn = gr.Button("🔍 Upscale Image", variant="primary", size="lg")
                        
                        # Main parameters (megapixels first, then scale_by)
                        with gr.Row():
                            megapixels = gr.Slider(
                                label="Megapixels",
                                value=1.0,
                                minimum=0.5,
                                maximum=2.0,
                                step=0.1,
                                info="Scales input to reasonable size for diffusion"
                            )
                            scale_by = gr.Slider(
                                label="Scale Factor",
                                value=1.5,
                                minimum=1.1,
                                maximum=2.0,
                                step=0.1,
                                info="The actual upscale multiplier"
                            )
                        
                        # Seed controls
                        with gr.Row():
                            seed = gr.Number(
                                label="Seed",
                                value=new_random_seed(),
                                minimum=0,
                                step=1,
                                scale=2
                            )
                            randomize_seed = gr.Checkbox(
                                label="🎲 Random",
                                value=True,
                                scale=0,
                                min_width=100
                            )
                        
                        # Advanced Settings Accordion
                        with gr.Accordion("⚙️ Advanced Settings", open=False):
                            gr.Markdown("*FlowMatchEulerDiscreteScheduler parameters for fine-tuning*")
                            
                            with gr.Group():
                                with gr.Row():
                                    steps = gr.Slider(
                                        label="Steps",
                                        value=10,
                                        minimum=5,
                                        maximum=20,
                                        step=1,
                                        info="Total diffusion steps for sigma schedule"
                                    )
                                    cfg = gr.Slider(
                                        label="CFG",
                                        value=1.0,
                                        minimum=1.0,
                                        maximum=5.0,
                                        step=0.1,
                                        info="Classifier-free guidance scale"
                                    )
                            
                            with gr.Group():
                                with gr.Row():
                                    start_at_step = gr.Slider(
                                        label="Start Step",
                                        value=5,
                                        minimum=0,
                                        maximum=20,
                                        step=1,
                                        info="Starting step index (0 = beginning)"
                                    )
                                    end_at_step = gr.Slider(
                                        label="End Step",
                                        value=10,
                                        minimum=0,
                                        maximum=20,
                                        step=1,
                                        info="Ending step index (higher than steps = use all)"
                                    )
                            
                            with gr.Group():
                                with gr.Row():
                                    shift = gr.Slider(
                                        label="Shift",
                                        value=3.0,
                                        minimum=1.0,
                                        maximum=10.0,
                                        step=0.5,
                                        info="Global timestep shift. Z-Image-Turbo uses 3.0"
                                    )
                                    sampler_name = gr.Dropdown(
                                        label="Sampler",
                                        choices=samplers,
                                        value="dpmpp_sde" if "dpmpp_sde" in samplers else samplers[0],
                                        info="Sampling algorithm"
                                    )
                            
                            gr.Markdown("##### Scheduler Fine-Tuning")
                            with gr.Group():
                                with gr.Row():
                                    base_shift = gr.Slider(
                                        label="Base Shift",
                                        value=0.5,
                                        minimum=0.0,
                                        maximum=2.0,
                                        step=0.01,
                                        info="Stabilizes generation. Higher = more consistent"
                                    )
                                    max_shift = gr.Slider(
                                        label="Max Shift",
                                        value=1.15,
                                        minimum=0.5,
                                        maximum=3.0,
                                        step=0.01,
                                        info="Maximum variation. Higher = more stylized"
                                    )
                            
                            with gr.Group():
                                with gr.Row():
                                    use_karras_sigmas = gr.Dropdown(
                                        label="Karras Sigmas",
                                        choices=["disable", "enable"],
                                        value="disable",
                                        info="Karras noise schedule for smoother results"
                                    )
                                    stochastic_sampling = gr.Dropdown(
                                        label="Stochastic Sampling",
                                        choices=["disable", "enable"],
                                        value="disable",
                                        info="Adds randomness (like ancestral samplers)"
                                    )
                            
                        # Model Selection
                        with gr.Accordion("Model Selection", open=True):
                            with gr.Row():
                                use_gguf = gr.Radio(
                                    choices=[("Standard", False), ("GGUF", True)],
                                    value=default_gguf_mode,
                                    label="Mode",
                                    scale=1,
                                    info="GGUF uses less VRAM"
                                )
                            with gr.Group():
                                default_diffusion = DEFAULT_DIFFUSION_GGUF if default_gguf_mode else DEFAULT_DIFFUSION
                                default_clip = DEFAULT_CLIP_GGUF if default_gguf_mode else DEFAULT_CLIP
                                unet_name = gr.Dropdown(
                                    label="Diffusion Model",
                                    choices=diffusion_models,
                                    value=get_default_model(diffusion_models, default_diffusion),
                                    info="UNet/DiT model for generation"
                                )
                                clip_name = gr.Dropdown(
                                    label="Text Encoder (CLIP)",
                                    choices=clip_models,
                                    value=get_default_model(clip_models, default_clip),
                                    info="Text encoder for prompt processing"
                                )
                                vae_name = gr.Dropdown(
                                    label="VAE",
                                    choices=vae_models,
                                    value=get_default_model(vae_models, DEFAULT_VAE),
                                    info="Variational autoencoder for image encoding/decoding"
                                )
                        
                        # Custom node status section (moved to bottom)
                        with gr.Accordion("📦 Custom Node Status", open=True):
                            node_status = gr.Textbox(
                                label="ComfyUI-EulerDiscreteScheduler",
                                value="✓ Installed" if is_installed else "⚠️ Not installed - Required for upscale workflow",
                                interactive=False
                            )
                            
                            with gr.Row():
                                install_btn = gr.Button(
                                    "📥 Install Custom Node",
                                    visible=not is_installed,
                                    variant="primary"
                                )
                                update_btn = gr.Button(
                                    "🔄 Update Custom Node",
                                    visible=is_installed
                                )
                    
                    # ===== RIGHT COLUMN - Output =====
                    with gr.Column(scale=1):
                        # ImageSlider output
                        output_slider = gr.ImageSlider(
                            label="Before / After",
                            type="filepath",
                            elem_classes="image-window",
                            show_download_button=False
                        )
                        
                        # Output controls
                        with gr.Row():
                            save_btn = gr.Button("💾 Save", size="sm", variant="primary")
                            open_folder_btn = gr.Button("📂 Open Folder", size="sm")
                            autosave = gr.Checkbox(label="Auto-save", value=False)
                        
                        # Status textbox
                        status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            show_label=False,
                            lines=1
                        )
                        
                        # System monitor
                        with gr.Row():
                            with gr.Column(scale=1, min_width=200):
                                gpu_monitor = gr.Textbox(
                                    value="Loading...",
                                    lines=4.5,
                                    container=False,
                                    interactive=False,
                                    show_label=False,
                                    elem_classes="monitor-box gpu-monitor"
                                )
                            with gr.Column(scale=1, min_width=200):
                                cpu_monitor = gr.Textbox(
                                    value="Loading...",
                                    lines=4,
                                    container=False,
                                    interactive=False,
                                    show_label=False,
                                    elem_classes="monitor-box cpu-monitor"
                                )
                        
                        # Hidden state for result path
                        result_path_state = gr.State(value=None)
                        original_path_state = gr.State(value=None)
        
        # ===== EVENT HANDLERS =====
        
        # Install button handler
        def on_install():
            success, msg = install_custom_node(
                custom_nodes_dir, CUSTOM_NODE_REPO, CUSTOM_NODE_NAME
            )
            if success:
                return (
                    msg,
                    gr.update(visible=False),
                    gr.update(visible=True)
                )
            return msg, gr.update(), gr.update()
        
        install_btn.click(
            fn=on_install,
            outputs=[node_status, install_btn, update_btn]
        )
        
        # Update button handler
        def on_update():
            success, msg = update_custom_node(custom_nodes_dir, CUSTOM_NODE_NAME)
            return msg
        
        update_btn.click(
            fn=on_update,
            outputs=[node_status]
        )
        
        # Update end_at_step max when steps changes
        def update_step_ranges(steps_val):
            return (
                gr.update(maximum=steps_val),
                gr.update(maximum=steps_val, value=min(steps_val, 10))
            )
        
        steps.change(
            fn=update_step_ranges,
            inputs=[steps],
            outputs=[start_at_step, end_at_step]
        )
        
        # Upscale button - async generator pattern (avoids progress spinner issue)
        async def upscale_main(
            img, prompt_text, seed_val, randomize, mp, scale, steps_val,
            start_step, end_step, shift_val, cfg_val, sampler,
            is_gguf, unet, clip, vae,
            base_shift_val, max_shift_val, karras, stochastic, auto
        ):
            async for result in experimental_upscale(
                services, img, prompt_text, seed_val, randomize, mp, scale,
                steps_val, start_step, end_step, shift_val, cfg_val, sampler,
                is_gguf, unet, clip, vae,
                base_shift_val, max_shift_val, karras, stochastic, auto
            ):
                slider_tuple, status_msg, actual_seed, res_path = result
                yield slider_tuple, status_msg, actual_seed, res_path, img
        
        upscale_btn.click(
            fn=upscale_main,
            inputs=[
                input_image, prompt, seed, randomize_seed, megapixels, scale_by,
                steps, start_at_step, end_at_step, shift, cfg, sampler_name,
                use_gguf, unet_name, clip_name, vae_name,
                base_shift, max_shift, use_karras_sigmas, stochastic_sampling, autosave
            ],
            outputs=[output_slider, status, seed, result_path_state, original_path_state]
        )
        
        # Update model dropdowns when GGUF mode changes
        def update_model_dropdowns(is_gguf):
            """Refresh model dropdowns based on GGUF mode."""
            new_diffusion = get_models_by_mode(diffusion_dir, is_gguf, DEFAULT_DIFFUSION, DEFAULT_DIFFUSION_GGUF, ZIMAGE_FILTERS["diffusion"])
            new_clip = get_models_by_mode(text_encoders_dir, is_gguf, DEFAULT_CLIP, DEFAULT_CLIP_GGUF, ZIMAGE_FILTERS["text_encoder"])
            
            default_diff = DEFAULT_DIFFUSION_GGUF if is_gguf else DEFAULT_DIFFUSION
            default_te = DEFAULT_CLIP_GGUF if is_gguf else DEFAULT_CLIP
            
            return (
                gr.update(choices=new_diffusion, value=get_default_model(new_diffusion, default_diff)),
                gr.update(choices=new_clip, value=get_default_model(new_clip, default_te))
            )
        
        use_gguf.change(
            fn=update_model_dropdowns,
            inputs=[use_gguf],
            outputs=[unet_name, clip_name]
        )
        
        # Save button handler
        def on_save(res_path, orig_path):
            if res_path is None:
                return "❌ No image to save"
            try:
                saved_path = save_experimental_output(res_path, orig_path, outputs_dir)
                return f"✓ Saved to {Path(saved_path).name}"
            except Exception as e:
                return f"❌ Save failed: {str(e)}"
        
        save_btn.click(
            fn=on_save,
            inputs=[result_path_state, original_path_state],
            outputs=[status]
        )
        
        # Open folder button
        def open_experimental_folder():
            open_folder(experimental_dir)
        
        open_folder_btn.click(fn=open_experimental_folder)
        
        # System Monitor
        def update_monitor():
            if services.system_monitor:
                gpu_info, cpu_info = services.system_monitor.get_system_info()
                return gpu_info, cpu_info
            return "N/A", "N/A"
        
        monitor_timer = gr.Timer(2, active=True)
        monitor_timer.tick(fn=update_monitor, outputs=[gpu_monitor, cpu_monitor])
    
    return tab
