from typing import Text
from pyparsing import *
from odinson.ruleutils import config
from odinson.ruleutils.queryast import *

__all__ = [
    "parse_odinson_query",
    "parse_surface",
    "parse_traversal",
]

# punctuation
comma = Literal(",").suppress()
equals = Literal("=").suppress()
vbar = Literal("|").suppress()
lt = Literal("<").suppress()
gt = Literal(">").suppress()
at = Literal("@").suppress()
ampersand = Literal("&").suppress()
open_curly = Literal("{").suppress()
close_curly = Literal("}").suppress()
open_parens = Literal("(").suppress()
close_parens = Literal(")").suppress()
open_bracket = Literal("[").suppress()
close_bracket = Literal("]").suppress()

# literal values
surface_hole = config.SURFACE_HOLE_GLYPH
traversal_hole = config.TRAVERSAL_HOLE_GLYPH
query_hole = config.QUERY_HOLE_GLYPH
number = Word(nums).setParseAction(lambda t: int(t[0]))
identifier = Word(alphas + "_", alphanums + "_")
single_quoted_string = QuotedString("'", unquoteResults=True, escChar="\\")
double_quoted_string = QuotedString('"', unquoteResults=True, escChar="\\")
quoted_string = single_quoted_string | double_quoted_string
string = identifier | quoted_string

# number to the left of the comma {n,}
quant_range_left = open_curly + number + comma + close_curly
quant_range_left.setParseAction(lambda t: (t[0], None))
# number to the right of the comma {,m}
quant_range_right = open_curly + comma + number + close_curly
quant_range_right.setParseAction(lambda t: (0, t[0]))
# numbers on both sides of the comma {n,m}
quant_range_both = open_curly + number + comma + number + close_curly
quant_range_both.setParseAction(lambda t: (t[0], t[1]))
# no number either side of the comma {,}
quant_range_neither = open_curly + comma + close_curly
quant_range_neither.setParseAction(lambda t: (0, None))
# range {n,m}
quant_range = (
        quant_range_left | quant_range_right | quant_range_both | quant_range_neither
)
# repetition {n}
quant_rep = open_curly + number + close_curly
quant_rep.setParseAction(lambda t: (t[0], t[0]))
# quantifier operator
quant_op = oneOf("? * +")
quant_op.setParseAction(
    lambda t: (0, 1) if t[0] == "?" else (0, None) if t[0] == "*" else (1, None)
)
# any quantifier
quantifier = quant_op | quant_range | quant_rep

# a hole that can take the place of a matcher
hole_matcher = Literal(surface_hole).setParseAction(lambda t: HoleMatcher())
# a matcher that compares tokens to a string (t[0])
exact_matcher = string.setParseAction(lambda t: ExactMatcher(t[0]))
# any matcher
matcher = hole_matcher | exact_matcher

# a hole that can take the place of a token constraint
hole_constraint = Literal(surface_hole).setParseAction(lambda t: HoleConstraint())

# a constraint of the form `f=v` means that only tokens
# that have a field `f` with a corresponding value of `v`
# can be accepted
field_constraint = matcher + equals + matcher
field_constraint.setParseAction(lambda t: FieldConstraint(*t))

# forward declaration, defined below
or_constraint = Forward()

# an expression that represents a single constraint
atomic_constraint = (
        field_constraint | hole_constraint | open_parens + or_constraint + close_parens
)

# a constraint that may or may not be negated
not_constraint = Optional("!") + atomic_constraint
not_constraint.setParseAction(lambda t: NotConstraint(t[1]) if len(t) > 1 else t[0])

# one or two constraints ANDed together
and_constraint = Forward()
and_constraint << (not_constraint + Optional(ampersand + and_constraint))
and_constraint.setParseAction(lambda t: AndConstraint(*t) if len(t) == 2 else t[0])

# one or two constraints ORed together
or_constraint << (and_constraint + Optional(vbar + or_constraint))
or_constraint.setParseAction(lambda t: OrConstraint(*t) if len(t) == 2 else t[0])

# a hole that can take the place of a surface query
hole_surface = Literal(surface_hole).setParseAction(lambda t: HoleSurface())

# a token constraint surrounded by square brackets
token_constraint = open_bracket + or_constraint + close_bracket
token_constraint.setParseAction(lambda t: TokenSurface(t[0]))

# an unconstrained token
token_wildcard = open_bracket + close_bracket
token_wildcard.setParseAction(lambda t: WildcardSurface())

# a token pattern
token_surface = token_wildcard | token_constraint

# forward declaration, defined below
or_surface = Forward()

# an entity or event mention
mention_surface = at + matcher
mention_surface.setParseAction(lambda t: MentionSurface(t[0]))

# an expression that represents a single query
atomic_surface = (
        hole_surface
        | token_surface
        | mention_surface
        | open_parens + or_surface + close_parens
)

# a query with an optional quantifier
repeat_surface = atomic_surface + Optional(quantifier)
repeat_surface.setParseAction(
    lambda t: RepeatSurface(t[0], *t[1]) if len(t) > 1 else t[0]
)

# one or two queries that must match consecutively
concat_surface = Forward()
concat_surface << (repeat_surface + Optional(concat_surface))
concat_surface.setParseAction(lambda t: ConcatSurface(*t) if len(t) == 2 else t[0])

# one or two queries ORed together
or_surface << (concat_surface + Optional(vbar + or_surface))
or_surface.setParseAction(lambda t: OrSurface(*t) if len(t) == 2 else t[0])

# a hole that can take the place of a traversal
hole_traversal = Literal(traversal_hole).setParseAction(lambda t: HoleTraversal())

# labeled incoming edge
incoming_label = lt + matcher
incoming_label.setParseAction(lambda t: IncomingLabelTraversal(t[0]))

# any incoming edge
incoming_wildcard = Literal("<<")
incoming_wildcard.setParseAction(lambda t: IncomingWildcardTraversal())

# an incoming edge
incoming_traversal = incoming_label | incoming_wildcard

# labeled outgoing edge
outgoing_label = gt + matcher
outgoing_label.setParseAction(lambda t: OutgoingLabelTraversal(t[0]))

# any outgoing edge
outgoing_wildcard = Literal(">>")
outgoing_wildcard.setParseAction(lambda t: OutgoingWildcardTraversal())

# an outgoing edge
outgoing_traversal = outgoing_label | outgoing_wildcard

# forward declaration, defined below
or_traversal = Forward()

# an expression that represents a single traversal
atomic_traversal = (
        hole_traversal
        | incoming_traversal
        | outgoing_traversal
        | open_parens + or_traversal + close_parens
)

# a traversal with an optional quantifier
repeat_traversal = atomic_traversal + Optional(quantifier)
repeat_traversal.setParseAction(
    lambda t: RepeatTraversal(t[0], *t[1]) if len(t) > 1 else t[0]
)

# one or two traversals that must match consecutively
concat_traversal = Forward()
concat_traversal << (repeat_traversal + Optional(concat_traversal))
concat_traversal.setParseAction(lambda t: ConcatTraversal(*t) if len(t) == 2 else t[0])

# one or two traversals ORed together
or_traversal << (concat_traversal + Optional(vbar + or_traversal))
or_traversal.setParseAction(lambda t: OrTraversal(*t) if len(t) == 2 else t[0])

# a hole that can take the place of a hybrid query
hole_query = Literal(query_hole).setParseAction(lambda t: HoleQuery())

# forward declaration, defined below
odinson_query = Forward()

# a single surface or a hybrid (surface, traversal, surface)
hybrid_query = Forward()
hybrid_query << (or_surface + Optional(or_traversal + odinson_query))
hybrid_query.setParseAction(lambda t: HybridQuery(*t) if len(t) == 3 else t[0])

odinson_query << (hole_query | hybrid_query)

# the top symbol of our grammar
top = LineStart() + odinson_query + LineEnd()


def parse_odinson_query(pattern: Text) -> AstNode:
    """Gets a string and returns the corresponding AST."""
    return top.parseString(pattern)[0]


def parse_surface(pattern: Text) -> Surface:
    """Gets a string and returns the corresponding surface pattern."""
    return or_surface.parseString(pattern)[0]


def parse_traversal(pattern: Text) -> Traversal:
    """Gets a string and returns the corresponding graph traversal."""
    return or_traversal.parseString(pattern)[0]


def parse_innermost_substitution(s: Text, container = None):
    """Gets a string and returns the corresponding AstNode element"""
    if s == "\u25a1=\u25a1":
        ins = FieldConstraint(HoleMatcher(), HoleMatcher())
    elif s == "!\u25a1":
        ins = NotConstraint(HoleConstraint())
    else:
        try:
            ins = parse_odinson_query(s)
        except ParseException:
            if s == '"\\""':
                ins = ExactMatcher('\"')
            elif s == '"\\\\"':
                ins = ExactMatcher('\\')
            else:
                ins = ExactMatcher(s.strip("\""))

    # Fix or constraints
    if container and isinstance(container, Constraint) and type(ins) == OrSurface:
        ins = OrConstraint(lhs=ins.lhs, rhs=ins.rhs)
    return ins
