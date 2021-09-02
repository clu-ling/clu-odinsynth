import re

HOLE_GLYPH = "\u25a1"  # WHITE SQUARE
IDENT_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]*$")
