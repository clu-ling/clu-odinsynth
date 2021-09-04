import re

SURFACE_HOLE_GLYPH = "\u25a1"    # WHITE SQUARE
TRAVERSAL_HOLE_GLYPH = "\u25c7"  # WHITE DIAMOND
QUERY_HOLE_GLYPH = "\u25cb"      # WHITE CIRCLE

IDENT_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]*$")

SYNTAX_FIELD = "dependencies"
