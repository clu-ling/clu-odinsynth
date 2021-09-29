import re
from typing import Dict, List, Text

# type alias
Vocabularies = Dict[Text, List[Text]]

SURFACE_HOLE_GLYPH = "\u25a1"  # WHITE SQUARE
TRAVERSAL_HOLE_GLYPH = "\u25b7"  # WHITE RIGHT-POINTING TRIANGLE
QUERY_HOLE_GLYPH = "\u2344"  # APL FUNCTIONAL SYMBOL QUAD GREATER-THAN

IDENT_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

SYNTAX_FIELD = "dependencies"
ENTITY_FIELD = "entity"
EXCLUDE_FIELDS = set([SYNTAX_FIELD, ENTITY_FIELD])
