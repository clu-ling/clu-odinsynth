import re

SURFACE_HOLE_GLYPH = "\u25a1"    # WHITE SQUARE
TRAVERSAL_HOLE_GLYPH = "\u25b7"  # WHITE RIGHT-POINTING TRIANGLE
QUERY_HOLE_GLYPH = "\u2344"      # APL FUNCTIONAL SYMBOL QUAD GREATER-THAN

IDENT_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]*$")

SYNTAX_FIELD = "dependencies"
