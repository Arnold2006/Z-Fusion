"""
CLIPTextEncodeWithWildcards + PreviewImageWithMeta nodes

Wildcard substitution:
- __name__ : Replaced with random line from wildcards/name.txt (seed-based)
- {a|b|c}  : Replaced with random choice from options (seed-based)

Wildcards are deterministic per seed, so the same seed + prompt = same result.

Wire resolved_prompt STRING output → PreviewImageWithMeta to guarantee the
resolved prompt is embedded in PNG metadata regardless of execution order.
"""

import json
import os
import random
import re

import numpy as np
import folder_paths
from PIL import Image
from PIL.PngImagePlugin import PngInfo


class CLIPTextEncodeWithWildcards:
    """CLIP Text Encode with wildcard and inline option substitution."""

    def __init__(self):
        self.wildcards_dir = os.path.join(folder_paths.base_path, "wildcards")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "clip": ("CLIP",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "resolved_prompt")
    FUNCTION = "encode"
    CATEGORY = "conditioning"

    def read_wildcard_file(self, filename: str) -> list[str]:
        """Read lines from a wildcard text file."""
        filepath = os.path.join(self.wildcards_dir, filename)
        if not os.path.exists(filepath):
            return []

        with open(filepath, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        return lines

    def process_wildcards(self, text: str, seed: int) -> str:
        """
        Process all wildcards in text using seed-based random selection.

        Supports:
        - __name__ : Loads from wildcards/name.txt
        - {opt1|opt2|opt3} : Inline random selection
        """
        rng = random.Random(seed)

        wildcard_pattern = re.compile(r"__([a-zA-Z0-9_-]+)__")

        def replace_wildcard(match):
            name = match.group(1)
            filename = f"{name}.txt"
            lines = self.read_wildcard_file(filename)
            if not lines:
                print(f"[Wildcards] File not found or empty: {filename}")
                return match.group(0)
            return rng.choice(lines)

        text = wildcard_pattern.sub(replace_wildcard, text)

        inline_pattern = re.compile(r"\{([^{}]*\|[^{}]*)\}")

        def replace_inline(match):
            options = [opt.strip() for opt in match.group(1).split("|") if opt.strip()]
            if not options:
                return ""
            return rng.choice(options)

        text = inline_pattern.sub(replace_inline, text)

        return text

    def encode(self, clip, text: str, seed: int):
        """Process wildcards and encode text with CLIP."""
        processed_text = self.process_wildcards(text, seed)

        if processed_text != text:
            print(f"[Wildcards] Processed: {processed_text[:200]}{'...' if len(processed_text) > 200 else ''}")

        tokens = clip.tokenize(processed_text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)

        return ([[cond, {"pooled_output": pooled}]], processed_text)


class PreviewImageWithMeta:
    """
    Drop-in replacement for PreviewImage that accepts resolved_prompt as an
    explicit STRING input.

    Why this is needed:
        ComfyUI schedules nodes purely by data dependency. The standard
        PreviewImage has no dependency on the wildcards node, so it can
        execute (and save the PNG) before resolved_prompt has been produced.
        By wiring resolved_prompt here as an explicit input, the scheduler
        is forced to run CLIPTextEncodeWithWildcards first — and we write
        the value directly into PngInfo at save time, bypassing extra_pnginfo
        mutation timing entirely.

    """

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + "".join(
            random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5)
        )
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                # Wire from CLIPTextEncodeWithWildcards `resolved_prompt` output.
                # Optional so this node also works in non-wildcard workflows.
                "resolved_prompt": ("STRING", {"forceInput": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_preview"
    OUTPUT_NODE = True
    CATEGORY = "image"

    def save_preview(
        self,
        images,
        resolved_prompt: str | None = None,
        prompt=None,
        extra_pnginfo=None,
    ):
        results = []

        for batch_idx, image in enumerate(images):
            # Convert tensor → PIL
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            # Build PNG metadata — mirrors what SaveImage does
            metadata = PngInfo()

            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))

            if extra_pnginfo is not None:
                for k, v in extra_pnginfo.items():
                    metadata.add_text(k, json.dumps(v))

            # Write resolved_prompt directly into PngInfo.
            # This is explicit and timing-independent — no extra_pnginfo mutation needed.
            if resolved_prompt is not None:
                metadata.add_text("resolved_prompt", json.dumps(resolved_prompt))

            # Save to temp dir (same behaviour as stock PreviewImage)
            filename_prefix = "ComfyUI" + self.prefix_append
            full_output_folder, filename, counter, subfolder, _ = (
                folder_paths.get_save_image_path(
                    filename_prefix, self.output_dir, img.width, img.height
                )
            )
            file = f"{filename}_{counter:05}_.png"
            img.save(
                os.path.join(full_output_folder, file),
                pnginfo=metadata,
                compress_level=self.compress_level,
            )

            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type,
            })

        return {"ui": {"images": results}}


NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodeWithWildcards": CLIPTextEncodeWithWildcards,
    "PreviewImageWithMeta": PreviewImageWithMeta,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTextEncodeWithWildcards": "CLIP Text Encode (Wildcards)",
    "PreviewImageWithMeta": "Preview Image (With Meta)",
}
