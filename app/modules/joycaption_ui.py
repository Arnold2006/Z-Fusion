"""
JoyCaption UI Module

Self-contained accordion UI for JoyCaption image captioning via ComfyUI.
Uses the JC_adv (Advanced) node + JC_ExtraOptions node.
Designed to be embedded in any tab via create_joycaption_ui() / run_joycaption().

Workflow: JoyCaption.json
  LoadImage → JC_ExtraOptions → JC_adv → STRING output
"""

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List

import gradio as gr
import httpx

if TYPE_CHECKING:
    from modules import SharedServices

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JC_adv node options (from JC.py / jc_data.json)
# ---------------------------------------------------------------------------

HF_MODELS = [
    "joycaption-beta-one",
    "joycaption-beta-one-fp8",
    "joycaption-alpha-two",
]

QUANTIZATION_MODES = [
    "Balanced (8-bit)",
    "Full Precision (bf16)",
    "Maximum Savings (4-bit)",
]

CAPTION_TYPES = [
    "Descriptive",
    "Descriptive (Casual)",
    "Straightforward",
    "Stable Diffusion Prompt",
    "MidJourney",
    "Danbooru tag list",
    "e621 tag list",
    "Rule34 tag list",
    "Booru-like tag list",
    "Art Critic",
    "Product Listing",
    "Social Media Post",
]

CAPTION_LENGTHS = ["any", "very short", "short", "medium", "long", "very long"]

MEMORY_MODES = ["Keep in Memory", "Clear After Run", "Global Cache"]

# ---------------------------------------------------------------------------
# Extra options — ordered exactly as JC_ExtraOptions node widget_values
# (key, display_label, default, tooltip)
# ---------------------------------------------------------------------------

EXTRA_OPTIONS: List[tuple] = [
    ("exclude_people_info",               "Exclude People Info",               False, "Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style)."),
    ("include_lighting",                  "Include Lighting",                  True,  "Include information about lighting."),
    ("include_camera_angle",              "Include Camera Angle",              True,  "Include information about camera angle."),
    ("include_watermark",                 "Include Watermark",                 False, "Include information about whether there is a watermark or not."),
    ("include_JPEG_artifacts",            "Include JPEG Artifacts",            False, "Include information about whether there are JPEG artifacts or not."),
    ("include_exif",                      "Include EXIF",                      False, "If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc."),
    ("exclude_sexual",                    "Exclude Sexual",                    False, "Do NOT include anything sexual; keep it PG."),
    ("exclude_image_resolution",          "Exclude Image Resolution",          False, "Do NOT mention the image's resolution."),
    ("include_aesthetic_quality",         "Include Aesthetic Quality",         False, "You MUST include information about the subjective aesthetic quality of the image from low to very high."),
    ("include_composition_style",         "Include Composition Style",         False, "Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry."),
    ("exclude_text",                      "Exclude Text",                      False, "Do NOT mention any text that is in the image."),
    ("specify_depth_field",               "Specify Depth Field",               False, "Specify the depth of field and whether the background is in focus or blurred."),
    ("specify_lighting_sources",          "Specify Lighting Sources",          False, "If applicable, mention the likely use of artificial or natural lighting sources."),
    ("do_not_use_ambiguous_language",     "Do Not Use Ambiguous Language",     False, "Do NOT use any ambiguous language."),
    ("include_nsfw",                      "Include NSFW",                      False, "Include whether the image is sfw, suggestive, or nsfw."),
    ("only_describe_most_important_elements", "Only Most Important Elements",  False, "ONLY describe the most important elements of the image."),
    ("do_not_include_artist_name_or_title",   "No Artist Name/Title",          False, "If it is a work of art, do not include the artist's name or the title of the work."),
    ("identify_image_orientation",        "Identify Image Orientation",        False, "Identify the image orientation (portrait, landscape, or square) and aspect ratio if obvious."),
    ("use_vulgar_slang_and_profanity",    "Use Vulgar Slang/Profanity",        False, "Use vulgar slang and profanity."),
    ("do_not_use_polite_euphemisms",      "No Polite Euphemisms",              False, "Do NOT use polite euphemisms—lean into blunt, casual phrasing."),
    ("include_character_age",             "Include Character Age",             False, "Include information about the ages of any people/characters when applicable."),
    ("include_camera_shot_type",          "Include Camera Shot Type",          False, "Mention whether the image depicts an extreme close-up, close-up, medium shot, wide shot, etc."),
    ("exclude_mood_feeling",              "Exclude Mood/Feeling",              True,  "Do not mention the mood/feeling/etc of the image."),
    ("include_camera_vantage_height",     "Include Camera Vantage Height",     False, "Explicitly specify the vantage height (eye-level, low-angle, bird's-eye, etc.)."),
    ("mention_watermark",                 "Mention Watermark",                 False, "If there is a watermark, you must mention it."),
    ("avoid_meta_descriptive_phrases",    "Avoid Meta Phrases",                False, "Avoid useless meta phrases like 'This image shows…', 'You are looking at...', etc."),
    ("refer_character_name",              "Refer by Character Name",           False, "If there is a person/character in the image you must refer to them as the name specified below."),
]

# Keys in order (used for workflow param mapping)
EXTRA_OPTION_KEYS = [opt[0] for opt in EXTRA_OPTIONS]


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class JoyCaptionComponents:
    """All UI components returned by create_joycaption_ui."""
    # Inputs
    image: gr.Image
    custom_prompt: gr.Textbox
    character_name: gr.Textbox
    character_name_row: gr.Row
    # JC_adv settings
    model: gr.Dropdown
    quantization: gr.Dropdown
    caption_type: gr.Dropdown
    caption_length: gr.Dropdown
    max_new_tokens: gr.Slider
    temperature: gr.Slider
    top_p: gr.Slider
    top_k: gr.Slider
    memory_mode: gr.Dropdown
    # Extra options checkboxes (ordered list matching EXTRA_OPTIONS)
    extra_option_checks: list
    # Output + controls
    output_text: gr.Textbox
    generate_btn: gr.Button
    send_to_prompt_btn: gr.Button
    status: gr.Textbox


# ---------------------------------------------------------------------------
# Create UI
# ---------------------------------------------------------------------------

def create_joycaption_ui(
    accordion_label: str = "🎨 JoyCaption",
    accordion_open: bool = False,
    show_image_input: bool = True,
) -> JoyCaptionComponents:
    """
    Create the JoyCaption accordion UI.

    Args:
        accordion_label: Label for the outer accordion
        accordion_open: Whether accordion starts open
        show_image_input: Whether to show the image input (False when the
                          host tab already has an image to pass in)

    Returns:
        JoyCaptionComponents with all UI components
    """
    with gr.Accordion(accordion_label, open=accordion_open):

        image = gr.Image(
            label="Input Image",
            type="filepath",
            visible=show_image_input,
            elem_classes="image-window",
        )

        custom_prompt = gr.Textbox(
            label="Custom Prompt (overrides Caption Type when set)",
            placeholder="Leave empty to use Caption Type below…",
            lines=2,
        )

        # --- JC_adv Settings ---
        with gr.Accordion("⚙️ Caption Settings", open=True):
            with gr.Row():
                caption_type = gr.Dropdown(
                    label="Caption Type",
                    choices=CAPTION_TYPES,
                    value="Descriptive",
                    info="Style of caption to generate",
                )
                caption_length = gr.Dropdown(
                    label="Length",
                    choices=CAPTION_LENGTHS,
                    value="any",
                    info="Target caption length",
                )
            with gr.Row():
                model = gr.Dropdown(
                    label="Model",
                    choices=HF_MODELS,
                    value="joycaption-beta-one",
                    info="Auto-downloads on first use",
                )
                quantization = gr.Dropdown(
                    label="Quantization",
                    choices=QUANTIZATION_MODES,
                    value="Balanced (8-bit)",
                    info="8-bit recommended for most GPUs",
                )
            with gr.Row():
                max_new_tokens = gr.Slider(
                    label="Max Tokens",
                    value=512,
                    minimum=64,
                    maximum=2048,
                    step=64,
                    info="Maximum tokens to generate",
                )
                temperature = gr.Slider(
                    label="Temperature",
                    value=0.6,
                    minimum=0.0,
                    maximum=2.0,
                    step=0.05,
                    info="Higher = more creative",
                )
            with gr.Row():
                top_p = gr.Slider(
                    label="Top P",
                    value=0.9,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    info="Nucleus sampling threshold",
                )
                top_k = gr.Slider(
                    label="Top K",
                    value=0,
                    minimum=0,
                    maximum=100,
                    step=1,
                    info="0 = disabled",
                )
            memory_mode = gr.Dropdown(
                label="Memory Management",
                choices=MEMORY_MODES,
                value="Keep in Memory",
                info="'Keep in Memory' = faster repeated runs; 'Clear After Run' = frees VRAM after each caption",
            )

        # --- Extra Options ---
        with gr.Accordion("🔧 Extra Options", open=False):
            gr.Markdown("*Fine-tune what the model includes or excludes in its caption.*")
            extra_option_checks = []
            # Render in 2-column rows
            opts = EXTRA_OPTIONS
            for i in range(0, len(opts), 2):
                with gr.Row():
                    for key, label, default, tooltip in opts[i:i+2]:
                        cb = gr.Checkbox(
                            label=label,
                            value=default,
                            info=tooltip,
                            scale=1,
                        )
                        extra_option_checks.append(cb)

            # Character name — shown/hidden by refer_character_name toggle
            with gr.Row(visible=False) as character_name_row:
                character_name = gr.Textbox(
                    label="Character Name",
                    placeholder="e.g. Alice, John Doe…",
                    lines=1,
                    scale=1,
                )

        # --- Output ---
        output_text = gr.Textbox(
            label="Generated Caption",
            lines=5,
            max_lines=12,
            interactive=True,
            placeholder="Caption will appear here…",
        )
        with gr.Row():
            generate_btn = gr.Button("🎨 Generate Caption", variant="primary", scale=3)
            send_to_prompt_btn = gr.Button("📝 Send to Prompt", size="sm", scale=1)
        status = gr.Textbox(
            label="",
            show_label=False,
            interactive=False,
            lines=1,
            max_lines=1,
        )

    # Wire refer_character_name toggle → show/hide character_name_row
    # refer_character_name is the last extra option (index -1)
    refer_cb = extra_option_checks[EXTRA_OPTION_KEYS.index("refer_character_name")]
    refer_cb.change(
        fn=lambda v: gr.update(visible=v),
        inputs=[refer_cb],
        outputs=[character_name_row],
    )

    return JoyCaptionComponents(
        image=image,
        custom_prompt=custom_prompt,
        character_name=character_name,
        character_name_row=character_name_row,
        model=model,
        quantization=quantization,
        caption_type=caption_type,
        caption_length=caption_length,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        memory_mode=memory_mode,
        extra_option_checks=extra_option_checks,
        output_text=output_text,
        generate_btn=generate_btn,
        send_to_prompt_btn=send_to_prompt_btn,
        status=status,
    )


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def get_joycaption_inputs(jc: JoyCaptionComponents) -> list:
    """
    Return ordered list of Gradio components for use as .click() inputs.

    Order matches the parameter list of run_joycaption():
      image, custom_prompt, character_name,
      model, quantization, caption_type, caption_length,
      max_new_tokens, temperature, top_p, top_k, memory_mode,
      *extra_option_checks  (27 booleans)
    """
    return [
        jc.image,
        jc.custom_prompt,
        jc.character_name,
        jc.model,
        jc.quantization,
        jc.caption_type,
        jc.caption_length,
        jc.max_new_tokens,
        jc.temperature,
        jc.top_p,
        jc.top_k,
        jc.memory_mode,
        *jc.extra_option_checks,
    ]


def get_joycaption_outputs(jc: JoyCaptionComponents) -> list:
    """Return [output_text, status] for use as .click() outputs."""
    return [jc.output_text, jc.status]


# ---------------------------------------------------------------------------
# Workflow runner
# ---------------------------------------------------------------------------

async def run_joycaption(
    services: "SharedServices",
    image: str,
    custom_prompt: str,
    character_name: str,
    model: str,
    quantization: str,
    caption_type: str,
    caption_length: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    memory_mode: str,
    # 27 extra option booleans (positional, matching EXTRA_OPTIONS order)
    *extra_flags: bool,
) -> tuple[str, str]:
    """
    Execute the JoyCaption workflow via ComfyKit.

    Returns (caption_text, status_message).
    """
    if image is None:
        return "", "❌ Please provide an input image"

    workflow_path = services.workflows_dir / "JoyCaption.json"
    if not workflow_path.exists():
        return "", "❌ JoyCaption.json workflow not found"

    # Build extra_options dict — keys match JC_ExtraOptions node input names
    extra_options = {
        key: bool(extra_flags[i])
        for i, key in enumerate(EXTRA_OPTION_KEYS)
    }

    params = {
        # LoadImage
        "image": image,
        # JC_adv
        "model": model,
        "quantization": quantization,
        "prompt_style": caption_type,
        "caption_length": caption_length,
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "custom_prompt": custom_prompt.strip() if custom_prompt else "",
        "memory_management": memory_mode,
        # JC_ExtraOptions
        "character_name": character_name.strip() if character_name else "",
        **extra_options,
    }

    try:
        result = await services.kit.execute(str(workflow_path), params)

        if result.status == "error":
            return "", f"❌ {result.msg}"

        # Caption comes back as a text output — ComfyKit returns it in result.text
        # or as the first item in result.texts depending on the kit version.
        caption = ""
        if hasattr(result, "texts") and result.texts:
            caption = result.texts[0]
        elif hasattr(result, "text") and result.text:
            caption = result.text
        elif hasattr(result, "images") and result.images:
            # Fallback: shouldn't happen for text output
            caption = str(result.images[0])

        if not caption:
            return "", "❌ No caption returned"

        time_str = f" | {result.duration:.1f}s" if result.duration else ""
        return caption, f"✓ Done{time_str}"

    except Exception as e:
        logger.error(f"JoyCaption error: {e}", exc_info=True)
        return "", f"❌ {str(e)}"


# ---------------------------------------------------------------------------
# Handler wiring helper
# ---------------------------------------------------------------------------

def setup_joycaption_handlers(
    jc: JoyCaptionComponents,
    services: "SharedServices",
    external_image: gr.Image = None,
    prompt_target: gr.Textbox = None,
):
    """
    Wire generate + send-to-prompt button handlers.

    Args:
        jc: JoyCaptionComponents from create_joycaption_ui()
        services: SharedServices instance
        external_image: If provided, use this image component instead of jc.image
        prompt_target: If provided, wire the Send to Prompt button to populate this textbox
    """
    image_input = external_image if external_image is not None else jc.image

    async def _generate(
        img, custom_p, char_name,
        mdl, quant, cap_type, cap_len,
        max_tok, temp, tp, tk, mem_mode,
        *flags
    ):
        return await run_joycaption(
            services, img, custom_p, char_name,
            mdl, quant, cap_type, cap_len,
            max_tok, temp, tp, tk, mem_mode,
            *flags,
        )

    inputs = [
        image_input,
        jc.custom_prompt,
        jc.character_name,
        jc.model,
        jc.quantization,
        jc.caption_type,
        jc.caption_length,
        jc.max_new_tokens,
        jc.temperature,
        jc.top_p,
        jc.top_k,
        jc.memory_mode,
        *jc.extra_option_checks,
    ]

    jc.generate_btn.click(
        fn=_generate,
        inputs=inputs,
        outputs=[jc.output_text, jc.status],
    )

    # Send to Prompt button — populates the target prompt textbox if wired
    if prompt_target is not None:
        jc.send_to_prompt_btn.click(
            fn=lambda text: (text, "✓ Sent to prompt"),
            inputs=[jc.output_text],
            outputs=[prompt_target, jc.status],
        )
    else:
        # No target — disable the button visually
        jc.send_to_prompt_btn.click(
            fn=lambda text: (text, "ℹ️ No prompt target wired"),
            inputs=[jc.output_text],
            outputs=[jc.output_text, jc.status],
        )
