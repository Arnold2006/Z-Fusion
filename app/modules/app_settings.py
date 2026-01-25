"""
App Settings Module

Provides the application-wide settings tab with output directory,
temp folder controls, UI theme selection, and wildcards management.
"""

import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import gradio as gr

from modules.wildcard_processor import process_wildcards

if TYPE_CHECKING:
    from modules import SharedServices

logger = logging.getLogger(__name__)

# Module metadata
TAB_ID = "app_settings"
TAB_LABEL = "🛠️ App Settings"
TAB_ORDER = 3

# Gradio temp directory (uses GRADIO_TEMP_DIR env var if set, else system temp)
GRADIO_TEMP_DIR = Path(os.environ.get("GRADIO_TEMP_DIR", tempfile.gettempdir()))

# Wildcards directory (inside ComfyUI - where the node reads from)
def get_wildcards_dir(services: "SharedServices") -> Path:
    """Get the wildcards directory path (inside ComfyUI)."""
    return services.app_dir / "comfyui" / "wildcards"


def get_wildcard_files(wildcards_dir: Path) -> list[str]:
    """Get list of wildcard .txt files."""
    if not wildcards_dir.exists():
        return []
    return sorted([f.stem for f in wildcards_dir.glob("*.txt")])


def read_wildcard_file(wildcards_dir: Path, name: str) -> str:
    """Read contents of a wildcard file."""
    filepath = wildcards_dir / f"{name}.txt"
    if not filepath.exists():
        return ""
    return filepath.read_text(encoding="utf-8")


def save_wildcard_file(wildcards_dir: Path, name: str, content: str) -> tuple[bool, str]:
    """Save contents to a wildcard file."""
    try:
        wildcards_dir.mkdir(parents=True, exist_ok=True)
        # Sanitize filename
        safe_name = "".join(c for c in name if c.isalnum() or c in "-_").strip()
        if not safe_name:
            return False, "❌ Invalid filename"
        filepath = wildcards_dir / f"{safe_name}.txt"
        filepath.write_text(content, encoding="utf-8")
        return True, f"✓ Saved {safe_name}.txt"
    except Exception as e:
        return False, f"❌ Failed to save: {e}"

# Available themes
BUILTIN_THEMES = ["Default", "Soft", "Monochrome", "Glass", "Base", "Ocean", "Origin", "Citrus"]
COMMUNITY_THEMES = {
    "Miku": "NoCrypt/miku",
    "Interstellar": "Nymbo/Interstellar",
    "xkcd": "gstaff/xkcd",
}
ALL_THEME_NAMES = BUILTIN_THEMES + list(COMMUNITY_THEMES.keys())


def clear_temp_folder() -> tuple[bool, str]:
    """Clear the Gradio temp folder. Returns (success, message)."""
    try:
        # Resolve to absolute path
        temp_path = GRADIO_TEMP_DIR.resolve()
        logger.info(f"Clearing temp folder: {temp_path}")
        
        if temp_path.exists():
            file_count = sum(1 for _ in temp_path.rglob("*") if _.is_file())
            shutil.rmtree(temp_path)
            temp_path.mkdir(parents=True, exist_ok=True)
            return True, f"✓ Cleared {file_count} files from {temp_path}"
        return True, f"✓ Temp folder empty ({temp_path})"
    except Exception as e:
        logger.warning(f"Failed to clear temp folder: {e}")
        return False, f"❌ Failed to clear temp: {e}"


def get_clear_temp_on_start(settings_manager) -> bool:
    """Get the clear temp on start setting."""
    return settings_manager.get("clear_temp_on_start", False)


def set_clear_temp_on_start(enabled: bool, settings_manager) -> None:
    """Save the clear temp on start setting."""
    settings_manager.set("clear_temp_on_start", enabled)


def open_folder(folder_path: Path):
    """Cross-platform folder opener."""
    folder_path.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        os.startfile(folder_path)
    elif sys.platform == "darwin":
        subprocess.run(["open", str(folder_path)])
    else:
        subprocess.run(["xdg-open", str(folder_path)])


def create_tab(services: "SharedServices") -> gr.TabItem:
    """
    Create the App Settings tab with output directory, temp folder controls,
    and UI theme selection.
    
    Args:
        services: SharedServices instance with all dependencies
        
    Returns:
        gr.TabItem containing the App Settings interface
    """
    # Get current outputs directory
    outputs_dir = services.get_outputs_dir()
    default_outputs_dir = services.app_dir / "outputs" / "z-image-fusion"
    
    # Get current theme
    current_theme = services.settings.get("ui_theme", "Default")
    
    with gr.TabItem(TAB_LABEL, id=TAB_ID) as tab:
        
        # === Appearance Accordion (Theme + Colors) ===
        with gr.Accordion("🎨 Appearance", open=False):
            valid_colors = ["purple", "blue", "coral", "teal"]
            with gr.Row():
                theme_dropdown = gr.Dropdown(
                    choices=ALL_THEME_NAMES,
                    value=current_theme,
                    label="UI Theme",
                    info="Requires app restart",
                    scale=1
                )
                analysis_color_scheme = gr.Dropdown(
                    label="Analysis Panel Color",
                    choices=valid_colors,
                    value=services.settings.get("analysis_color_scheme"),
                    scale=1
                )
            with gr.Row():
                theme_apply_btn = gr.Button("🎨 Apply Theme (requires restart)", variant="primary", size="sm")
                analysis_color_apply_btn = gr.Button("✨ Apply Panel Color", size="sm")
            
            gr.Markdown("---")
            gr.Markdown("**Output Gallery**")
            with gr.Row():
                gallery_height = gr.Slider(
                    label="Gallery Height",
                    value=services.settings.get("output_gallery_height", 600),
                    minimum=200,
                    maximum=800,
                    step=50,
                    info="Requires app restart",
                    scale=1
                )
                gallery_max_images = gr.Slider(
                    label="Max Images",
                    value=services.settings.get("output_gallery_max_images", 100),
                    minimum=20,
                    maximum=100,
                    step=10,
                    info="Requires app restart",
                    scale=1
                )
                gallery_settings_apply_btn = gr.Button("💾 Save", size="sm", scale=0)
        
        # === Wildcards Accordion ===
        wildcards_dir = get_wildcards_dir(services)
        with gr.Accordion("🎲 Wildcards", open=False):
            gr.Markdown("""
**Prompt Wildcards** — *Add variety to your prompts with random substitutions.*

**Syntax:**
- `__name__` → Replaced with random line from `wildcards/name.txt`
- `{option1|option2|option3}` → Random inline selection

**Example prompt:**
```
A __camera__ of a {man|woman} with __haircolor__ __hairstyle__ and __eyecolor__, wearing __outfit__, in a __location__, __lighting__, __mood__
```
""")
            with gr.Row():
                wildcard_file_dropdown = gr.Dropdown(
                    choices=[""] + get_wildcard_files(wildcards_dir),
                    value="",
                    label="Edit Wildcard File",
                    scale=2
                )
                wildcard_refresh_btn = gr.Button("🔄", size="sm", scale=0)
                wildcard_open_folder_btn = gr.Button("📂 Open Folder", size="sm", scale=0)
            
            wildcard_editor = gr.Textbox(
                label="Contents (one item per line, # for comments)",
                lines=8,
                placeholder="Select a file above to edit, or enter a new filename below to create one..."
            )
            
            with gr.Row():
                wildcard_new_name = gr.Textbox(
                    label="New File Name",
                    placeholder="e.g. emotion",
                    scale=2
                )
                wildcard_save_btn = gr.Button("💾 Save", variant="primary", size="sm", scale=0)
            
            gr.Markdown("---")
            gr.Markdown("**Prompt Tester** — *Preview how wildcards resolve*")
            gr.Markdown("*The seed determines which wildcard options are selected. Same wildcard prompt + seed = same result.*", elem_classes=["info-text"])
            with gr.Row():
                wildcard_test_input = gr.Textbox(
                    label="Test Prompt",
                    placeholder="A __camera__ of a {man|woman} with __eyecolor__...",
                    lines=2,
                    scale=3
                )
                wildcard_test_seed = gr.Number(
                    label="Seed",
                    info="contains resolved prompt",
                    value=random.randint(0, 999999999),
                    minimum=0,
                    scale=1
                )
            wildcard_test_output = gr.Textbox(
                label="Resolved Prompt",
                lines=2,
                interactive=False
            )
            with gr.Row():
                wildcard_test_btn = gr.Button("🎲 Test", variant="primary", size="sm")
                wildcard_send_btn = gr.Button("📤 Send to Z-Image (unchecks 🎲 seed)", size="sm")
        
        # === Storage Accordion (Output Dir + Temp) ===
        with gr.Accordion("📁 Storage", open=False):
            gr.Markdown("**Output Directory** — *Where generated images and upscaled videos are saved.*")
            with gr.Row():
                app_outputs_dir = gr.Textbox(
                    value=str(outputs_dir),
                    placeholder="Leave empty for default",
                    show_label=False,
                    scale=3
                )
                app_outputs_browse_btn = gr.Button("📂", size="sm", scale=0)
            with gr.Row():
                app_outputs_save_btn = gr.Button("💾 Save Path", variant="primary", size="sm")
                app_outputs_reset_btn = gr.Button("↩️ Reset to Default", size="sm")
            gr.Markdown(f"*Default: `{default_outputs_dir}`*")
            
            gr.Markdown("---")
            gr.Markdown("**Temp Folder** — *Gradio stores previews and cached images here.*")
            with gr.Row():
                clear_temp_on_start = gr.Checkbox(
                    label="Clear temp folder on app start",
                    value=get_clear_temp_on_start(services.settings)
                )
                clear_temp_btn = gr.Button("🗑️ Clear Now", size="sm")
        
        # Status textbox outside accordions
        app_settings_status = gr.Textbox(label="", interactive=False, show_label=False, lines=1)
        
        # ===== EVENT HANDLERS =====
        
        def save_outputs_dir(path_str):
            """Save custom outputs directory."""
            path_str = path_str.strip()
            
            if not path_str:
                # Empty = use default
                settings = services.settings.load()
                if "outputs_dir" in settings:
                    del settings["outputs_dir"]
                    services.settings.save(settings)
                new_outputs_dir = services.get_outputs_dir()
                return f"✓ Reset to default: {new_outputs_dir}"
            
            path = Path(path_str)
            if not path.is_absolute():
                return "❌ Please enter an absolute path (e.g. C:\\Users\\...)"
            
            try:
                path.mkdir(parents=True, exist_ok=True)
                services.settings.set("outputs_dir", str(path))
                return f"✓ Saved: {path}"
            except Exception as e:
                return f"❌ Invalid path: {e}"
        
        def reset_outputs_dir():
            """Reset outputs directory to default."""
            settings = services.settings.load()
            if "outputs_dir" in settings:
                del settings["outputs_dir"]
                services.settings.save(settings)
            new_outputs_dir = services.get_outputs_dir()
            return str(new_outputs_dir), f"✓ Reset to default"
        
        def browse_outputs_dir():
            """Open file dialog - returns current path (user manually pastes)."""
            # Gradio doesn't have native folder picker, so just open the current folder
            current_outputs = services.get_outputs_dir()
            open_folder(current_outputs)
            return f"📂 Opened current folder. Copy your desired path and paste above."
        
        app_outputs_save_btn.click(
            fn=save_outputs_dir,
            inputs=[app_outputs_dir],
            outputs=[app_settings_status]
        )
        
        app_outputs_reset_btn.click(
            fn=reset_outputs_dir,
            outputs=[app_outputs_dir, app_settings_status]
        )
        
        app_outputs_browse_btn.click(
            fn=browse_outputs_dir,
            outputs=[app_settings_status]
        )
        
        # Theme handlers
        def on_theme_apply(theme_name):
            if not theme_name:
                return "❌ Please select a theme first"
            services.settings.set("ui_theme", theme_name)
            return f"✓ Theme set to '{theme_name}'. Please restart the app and refresh the webui to apply."
        
        theme_apply_btn.click(
            fn=on_theme_apply,
            inputs=[theme_dropdown],
            outputs=[app_settings_status]
        )
        
        # Appearance handlers
        def on_analysis_color_apply(color):
            if not color:
                return "❌ Please select a color first"
            services.settings.set("analysis_color_scheme", color)
            return f"✓ Analysis color set to {color}"
        
        analysis_color_apply_btn.click(
            fn=on_analysis_color_apply,
            inputs=[analysis_color_scheme],
            outputs=[app_settings_status]
        )
        
        # Gallery settings handler
        def on_gallery_settings_apply(height, max_imgs):
            services.settings.set("output_gallery_height", int(height))
            services.settings.set("output_gallery_max_images", int(max_imgs))
            return f"✓ Gallery: {int(height)}px height, {int(max_imgs)} max images. Restart to apply."
        
        gallery_settings_apply_btn.click(
            fn=on_gallery_settings_apply,
            inputs=[gallery_height, gallery_max_images],
            outputs=[app_settings_status]
        )
        
        # Temp folder handlers
        def on_clear_temp_on_start_change(enabled):
            set_clear_temp_on_start(enabled, services.settings)
            return f"✓ Clear on start: {'enabled' if enabled else 'disabled'}"
        
        def on_clear_temp_now():
            success, msg = clear_temp_folder()
            return msg
        
        clear_temp_on_start.change(
            fn=on_clear_temp_on_start_change,
            inputs=[clear_temp_on_start],
            outputs=[app_settings_status]
        )
        
        clear_temp_btn.click(
            fn=on_clear_temp_now,
            outputs=[app_settings_status]
        )
        
        # Wildcards handlers
        def on_wildcard_file_select(filename):
            if not filename:
                return ""
            return read_wildcard_file(wildcards_dir, filename)
        
        def on_wildcard_refresh():
            files = [""] + get_wildcard_files(wildcards_dir)
            return gr.update(choices=files, value="")
        
        def on_wildcard_open_folder():
            wildcards_dir.mkdir(parents=True, exist_ok=True)
            open_folder(wildcards_dir)
        
        def on_wildcard_save(selected_file, new_name, content):
            # Use new_name if provided, otherwise use selected file
            filename = new_name.strip() if new_name.strip() else selected_file
            if not filename:
                return "❌ Please select a file or enter a new filename", gr.update(), gr.update()
            
            success, msg = save_wildcard_file(wildcards_dir, filename, content)
            
            # Refresh dropdown and clear new name field on success
            if success:
                files = get_wildcard_files(wildcards_dir)
                # Select the saved file
                saved_name = "".join(c for c in filename if c.isalnum() or c in "-_").strip()
                return msg, gr.update(choices=files, value=saved_name), ""
            return msg, gr.update(), gr.update()
        
        wildcard_file_dropdown.change(
            fn=on_wildcard_file_select,
            inputs=[wildcard_file_dropdown],
            outputs=[wildcard_editor]
        )
        
        wildcard_refresh_btn.click(
            fn=on_wildcard_refresh,
            outputs=[wildcard_file_dropdown]
        )
        
        wildcard_open_folder_btn.click(fn=on_wildcard_open_folder, outputs=[])
        
        wildcard_save_btn.click(
            fn=on_wildcard_save,
            inputs=[wildcard_file_dropdown, wildcard_new_name, wildcard_editor],
            outputs=[app_settings_status, wildcard_file_dropdown, wildcard_new_name]
        )
        
        # Prompt tester handlers
        def on_wildcard_test(prompt):
            if not prompt:
                return "", 0
            seed = random.randint(0, 999999999)
            resolved = process_wildcards(prompt, seed, wildcards_dir)
            return resolved, seed
        
        def on_wildcard_random_seed():
            return random.randint(0, 999999999)
        
        wildcard_test_btn.click(
            fn=on_wildcard_test,
            inputs=[wildcard_test_input],
            outputs=[wildcard_test_output, wildcard_test_seed]
        )
        
        # Register components for cross-module wiring (handled in app.py)
        services.inter_module.register_component("wildcard_test_input", wildcard_test_input)
        services.inter_module.register_component("wildcard_test_seed", wildcard_test_seed)
        services.inter_module.register_component("wildcard_send_btn", wildcard_send_btn)
    
    return tab
