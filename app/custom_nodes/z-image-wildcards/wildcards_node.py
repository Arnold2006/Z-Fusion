"""
CLIPTextEncodeWithWildcards node

Extends standard CLIP text encoding with wildcard substitution:
- __name__ : Replaced with random line from wildcards/name.txt (seed-based)
- {a|b|c}  : Replaced with random choice from options (seed-based)

Wildcards are deterministic per seed, so the same seed + prompt = same result.
"""

import os
import random
import re

import folder_paths


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
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
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
        
        # Process __wildcard__ patterns
        # Match __word__ or __word-word__ (alphanumeric and hyphens)
        wildcard_pattern = re.compile(r"__([a-zA-Z0-9_-]+)__")
        
        def replace_wildcard(match):
            name = match.group(1)
            filename = f"{name}.txt"
            lines = self.read_wildcard_file(filename)
            
            if not lines:
                print(f"[Wildcards] File not found or empty: {filename}")
                return match.group(0)  # Return original if not found
            
            return rng.choice(lines)
        
        text = wildcard_pattern.sub(replace_wildcard, text)
        
        # Process {option|option|option} patterns
        # Match {content} where content contains at least one |
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
        # Process wildcards
        processed_text = self.process_wildcards(text, seed)
        
        # Log the processed prompt for debugging
        if processed_text != text:
            print(f"[Wildcards] Processed: {processed_text[:200]}{'...' if len(processed_text) > 200 else ''}")
        
        # Standard CLIP encoding
        tokens = clip.tokenize(processed_text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        
        return ([[cond, {"pooled_output": pooled}]],)


NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodeWithWildcards": CLIPTextEncodeWithWildcards,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTextEncodeWithWildcards": "CLIP Text Encode (Wildcards)",
}
