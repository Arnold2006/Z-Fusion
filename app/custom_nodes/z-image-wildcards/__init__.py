"""
Z-Image Wildcards - Custom CLIP Text Encode with wildcard support

Supports:
- __wildcard__ syntax: Replaced with random line from wildcards/wildcard.txt
- {option1|option2|option3} syntax: Inline random selection
"""

from .wildcards_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
