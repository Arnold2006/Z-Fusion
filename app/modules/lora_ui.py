"""
LoRA UI Components

Reusable Gradio components for LoRA selection across modules.
Provides a consistent UI pattern for 3-slot LoRA selection with
enable checkboxes, dropdowns, and strength sliders.
"""

import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import gradio as gr

if TYPE_CHECKING:
    from modules import SharedServices

logger = logging.getLogger(__name__)

# Dummy lora filename (used when lora disabled - strength 0 bypasses it)
DUMMY_LORA = "none.safetensors"


def scan_loras(loras_dir: Path) -> list:
    """Scan loras directory for available LoRA files."""
    if not loras_dir.exists():
        return []
    loras = []
    for f in loras_dir.rglob("*.safetensors"):
        rel_path = str(f.relative_to(loras_dir))
        if rel_path != DUMMY_LORA:  # Exclude dummy
            loras.append(rel_path)
    return sorted(loras)


def ensure_dummy_lora(loras_dir: Path):
    """Create a minimal dummy lora file for disabled slots."""
    dummy_path = loras_dir / DUMMY_LORA
    if dummy_path.exists():
        return
    
    try:
        loras_dir.mkdir(parents=True, exist_ok=True)
        import torch
        from safetensors.torch import save_file
        save_file({"__placeholder__": torch.zeros(1)}, str(dummy_path))
        logger.info(f"Created dummy lora: {dummy_path}")
    except Exception as e:
        logger.warning(f"Could not create dummy lora: {e}")


def open_folder(folder_path: Path):
    """Cross-platform folder opener."""
    folder_path.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        os.startfile(folder_path)
    elif sys.platform == "darwin":
        subprocess.run(["open", str(folder_path)])
    else:
        subprocess.run(["xdg-open", str(folder_path)])


@dataclass
class LoraComponents:
    """Container for LoRA UI components returned by create_lora_ui."""
    # LoRA 1
    lora1_enabled: gr.Checkbox
    lora1_name: gr.Dropdown
    lora1_strength: gr.Slider
    # LoRA 2
    lora2_enabled: gr.Checkbox
    lora2_name: gr.Dropdown
    lora2_strength: gr.Slider
    # LoRA 3
    lora3_enabled: gr.Checkbox
    lora3_name: gr.Dropdown
    lora3_strength: gr.Slider
    # Buttons
    refresh_btn: gr.Button
    open_folder_btn: gr.Button


def create_lora_ui(loras_dir: Path, accordion_open: bool = False) -> LoraComponents:
    """
    Create the LoRA accordion UI with 3 slots.
    
    Args:
        loras_dir: Path to the loras directory
        accordion_open: Whether the accordion should be open by default
        
    Returns:
        LoraComponents dataclass with all UI components
    """
    # Ensure dummy lora exists
    ensure_dummy_lora(loras_dir)
    
    # Scan available loras
    loras = scan_loras(loras_dir)
    
    with gr.Accordion("🎨 LoRA", open=accordion_open):
        # LoRA 1
        with gr.Row():
            lora1_enabled = gr.Checkbox(label="", value=False, scale=0, min_width=30)
            lora1_name = gr.Dropdown(
                label="LoRA 1",
                choices=loras,
                value=None,
                interactive=True,
                scale=3,
                allow_custom_value=True
            )
            lora1_strength = gr.Slider(label="Strength", value=1.0, minimum=0.0, maximum=2.0, step=0.05, scale=1)
        
        # LoRA 2
        with gr.Row():
            lora2_enabled = gr.Checkbox(label="", value=False, scale=0, min_width=30)
            lora2_name = gr.Dropdown(
                label="LoRA 2",
                choices=loras,
                value=None,
                interactive=True,
                scale=3,
                allow_custom_value=True
            )
            lora2_strength = gr.Slider(label="Strength", value=1.0, minimum=0.0, maximum=2.0, step=0.05, scale=1)
        
        # LoRA 3
        with gr.Row():
            lora3_enabled = gr.Checkbox(label="", value=False, scale=0, min_width=30)
            lora3_name = gr.Dropdown(
                label="LoRA 3",
                choices=loras,
                value=None,
                interactive=True,
                scale=3,
                allow_custom_value=True
            )
            lora3_strength = gr.Slider(label="Strength", value=1.0, minimum=0.0, maximum=2.0, step=0.05, scale=1)
        
        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh", size="sm", scale=0)
            open_folder_btn = gr.Button("📂 Open LoRAs Folder", size="sm", scale=1)
        
        gr.Markdown("*⭐ Tip: Distilled models don't stack LoRAs well. Try lowering strength when using multiple.*")
    
    return LoraComponents(
        lora1_enabled=lora1_enabled,
        lora1_name=lora1_name,
        lora1_strength=lora1_strength,
        lora2_enabled=lora2_enabled,
        lora2_name=lora2_name,
        lora2_strength=lora2_strength,
        lora3_enabled=lora3_enabled,
        lora3_name=lora3_name,
        lora3_strength=lora3_strength,
        refresh_btn=refresh_btn,
        open_folder_btn=open_folder_btn,
    )


def setup_lora_handlers(lora_components: LoraComponents, loras_dir: Path):
    """
    Wire up event handlers for LoRA UI components.
    
    Args:
        lora_components: LoraComponents from create_lora_ui
        loras_dir: Path to the loras directory
    """
    # Refresh button - updates all 3 dropdowns
    def refresh_loras():
        loras = scan_loras(loras_dir)
        return (
            gr.update(choices=loras),
            gr.update(choices=loras),
            gr.update(choices=loras)
        )
    
    lora_components.refresh_btn.click(
        fn=refresh_loras,
        outputs=[
            lora_components.lora1_name,
            lora_components.lora2_name,
            lora_components.lora3_name
        ]
    )
    
    # Open folder button
    lora_components.open_folder_btn.click(
        fn=lambda: open_folder(loras_dir)
    )


def get_lora_params(
    lora1_enabled: bool, lora1_name: str, lora1_strength: float,
    lora2_enabled: bool, lora2_name: str, lora2_strength: float,
    lora3_enabled: bool, lora3_name: str, lora3_strength: float,
) -> dict:
    """
    Build LoRA params dict for workflow execution.
    
    Returns dict with lora1_name, lora1_strength, lora2_name, etc.
    Uses DUMMY_LORA with strength 0 for disabled slots.
    """
    return {
        "lora1_name": lora1_name if (lora1_enabled and lora1_name) else DUMMY_LORA,
        "lora1_strength": lora1_strength if (lora1_enabled and lora1_name) else 0,
        "lora2_name": lora2_name if (lora2_enabled and lora2_name) else DUMMY_LORA,
        "lora2_strength": lora2_strength if (lora2_enabled and lora2_name) else 0,
        "lora3_name": lora3_name if (lora3_enabled and lora3_name) else DUMMY_LORA,
        "lora3_strength": lora3_strength if (lora3_enabled and lora3_name) else 0,
    }


def get_lora_inputs(lora_components: LoraComponents) -> list:
    """
    Get list of LoRA input components for use in gr.Button.click() inputs.
    
    Returns list in order: [enabled1, name1, strength1, enabled2, name2, strength2, ...]
    """
    return [
        lora_components.lora1_enabled,
        lora_components.lora1_name,
        lora_components.lora1_strength,
        lora_components.lora2_enabled,
        lora_components.lora2_name,
        lora_components.lora2_strength,
        lora_components.lora3_enabled,
        lora_components.lora3_name,
        lora_components.lora3_strength,
    ]
