from pyparsing import *
from odinson.ruleutils.queryast import *
from typing import Text
from odinson.ruleutils import config


class QueryParser:
    """Parser for odinson-like language."""

    def __init__(self):
        self.parser = make_odinson_style_parser()

    def parse(self, pattern: Text) -> AstNode:
        """Gets a string and returns the corresponding AST."""
        return self.parser.parseString(pattern)[0]


def make_odinson_style_parser():
    """Returns a parser for the query language using pyparsing."""

    # punctuation
    comma = Literal(",").suppress()
    equals = Literal("=").suppress()
    vbar = Literal("|").suppress()
    ampersand = Literal("&").suppress()
    open_curly = Literal("{").suppress()
    close_curly = Literal("}").suppress()
    open_parens = Literal("(").suppress()
    close_parens = Literal(")").suppress()
    open_bracket = Literal("[").suppress()
    close_bracket = Literal("]").suppress()

    # literal values
    hole = config.HOLE_GLYPH
    number = Word(nums).setParseAction(lambda t: int(t[0]))
    identifier = Word(alphas + "_", alphanums + "_")
    quoted_string = QuotedString('"', unquoteResults=True, escChar="\\")
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
    hole_matcher = Literal(hole).setParseAction(lambda t: HoleMatcher())
    # a matcher that compares tokens to a string (t[0])
    exact_matcher = string.setParseAction(lambda t: ExactMatcher(t[0]))
    # any matcher
    matcher = hole_matcher | exact_matcher

    # a hole that can take the place of a token constraint
    hole_constraint = Literal(hole).setParseAction(lambda t: HoleConstraint())

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

    # a hole that can take the place of a query
    hole_query = Literal(hole).setParseAction(lambda t: HoleSurface())

    # a token constraint surrounded by square brackets
    token_surface = open_bracket + or_constraint + close_bracket
    token_surface.setParseAction(lambda t: TokenSurface(t[0]))

    # forward declaration, defined below
    or_surface = Forward()

    # an expression that represents a single query
    atomic_surface = (
        hole_query | token_surface | open_parens + or_surface + close_parens
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

    # the top symbol of our grammar
    basic_query = LineStart() + or_surface + LineEnd()

    return basic_query
