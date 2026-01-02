"""
Shared wildcard processing logic.

Used by both the Gradio UI (prompt tester) and the ComfyUI custom node.

Supports:
- __name__ : Replaced with random line from wildcards/name.txt (seed-based)
- {a|b|c}  : Replaced with random choice from options (seed-based)
"""

import os
import random
import re
from pathlib import Path


def read_wildcard_file(wildcards_dir: Path, filename: str) -> list[str]:
    """Read lines from a wildcard text file."""
    filepath = wildcards_dir / filename
    if not filepath.exists():
        return []
    
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return lines


def process_wildcards(text: str, seed: int, wildcards_dir: Path) -> str:
    """
    Process all wildcards in text using seed-based random selection.
    
    Args:
        text: Input text with wildcards
        seed: Seed for deterministic random selection
        wildcards_dir: Path to wildcards folder
    
    Returns:
        Processed text with wildcards resolved
    
    Supports:
    - __name__ : Loads from wildcards/name.txt
    - {opt1|opt2|opt3} : Inline random selection
    """
    rng = random.Random(seed)
    
    # Process __wildcard__ patterns
    wildcard_pattern = re.compile(r"__([a-zA-Z0-9_-]+)__")
    
    def replace_wildcard(match):
        name = match.group(1)
        filename = f"{name}.txt"
        lines = read_wildcard_file(wildcards_dir, filename)
        
        if not lines:
            return match.group(0)  # Return original if not found
        
        return rng.choice(lines)
    
    text = wildcard_pattern.sub(replace_wildcard, text)
    
    # Process {option|option|option} patterns
    inline_pattern = re.compile(r"\{([^{}]*\|[^{}]*)\}")
    
    def replace_inline(match):
        options = [opt.strip() for opt in match.group(1).split("|") if opt.strip()]
        if not options:
            return ""
        return rng.choice(options)
    
    text = inline_pattern.sub(replace_inline, text)
    
    return text
