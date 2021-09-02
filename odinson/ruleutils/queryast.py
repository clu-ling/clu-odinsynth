from __future__ import annotations

import re
import json
from typing import Dict, List, Optional, Text, Tuple, Type, Union
from odinson.ruleutils import config


__all__ = [
    "AstNode",
    "Matcher",
    "HoleMatcher",
    "ExactMatcher",
    "Constraint",
    "HoleConstraint",
    "FieldConstraint",
    "OrConstraint",
    "AndConstraint",
    "NotConstraint",
    "Surface",
    "HoleSurface",
    "TokenSurface",
    "ConcatSurface",
    "OrSurface",
    "RepeatSurface",
]


# type alias
Vocabularies = Dict[Text, List[Text]]


class AstNode:
    """The base class for all AST nodes."""

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self}>"

    def is_hole(self) -> bool:
        """Returns true if the node is a hole."""
        # most nodes are not holes,
        # so only the Hole* nodes need to override this
        return False

    def has_holes(self) -> bool:
        """Returns true if the pattern has one or more holes."""
        # most nodes need to override this to handle their children,
        # so the default implementation is intended for Hole* nodes
        return self.is_hole()

    def is_valid(self) -> bool:
        """Returns true if the pattern is valid, i.e., has no holes."""
        return not self.has_holes()

    def tokens(self) -> List[Text]:
        """Returns the pattern as a list of tokens."""
        # default implementation is intended for nodes that have no children
        return [Text(self)]

    def num_matcher_holes(self) -> int:
        """Returns the number of matcher holes in this pattern."""
        return 0

    def num_constraint_holes(self) -> int:
        """Returns the number of constraint holes in this pattern."""
        return 0

    def num_surface_holes(self) -> int:
        """Returns the number of surface holes in this pattern."""
        return 0

    def num_holes(self) -> int:
        """Returns the number of holes in this pattern."""
        return (
            self.num_matcher_holes()
            + self.num_constraint_holes()
            + self.num_surface_holes()
        )

    def expand_leftmost_hole(self, vocabularies: Vocabularies) -> List[AstNode]:
        """
        If the pattern has holes then it returns the patterns obtained
        by expanding the leftmost hole.  If there are no holes then it
        returns an empty list.
        """
        # default implementation is suitable for Matchers only
        return []

    def preorder_traversal(self) -> List[AstNode]:
        """Returns a list with all the nodes of the tree in preorder."""
        # default implementation is for nodes that have no children
        return [self]


# type alias
Types = Type[Union[AstNode, Tuple[AstNode]]]


def is_identifier(s: Text) -> bool:
    """returns true if the provided string is a valid identifier"""
    return config.IDENT_RE.match(s) is not None


def maybe_parens(node: AstNode, types: Types) -> str:
    """Converts node to string. Surrounds by parenthesis
    if node is subclass of provided types."""
    return f"({node})" if isinstance(node, types) else str(node)


def maybe_parens_tokens(node: AstNode, types: Types) -> List[Text]:
    """Converts node to list of tokens. Surrounds by parenthesis
    if node is subclass of provided types."""
    return ["(", *node.tokens(), ")"] if isinstance(node, types) else node.tokens()


def make_quantifier(min: int, max: Optional[int]) -> str:
    """Gets the desired minimum and maximum repetitions
    and returns the appropriate quantifier."""
    return "".join(make_quantifier_tokens(min, max))


def make_quantifier_tokens(min: int, max: Optional[int]) -> List[Text]:
    """Gets the desired minimum and maximum repetitions
    and returns the sequence of tokens corresponding
    to the appropriate quantifier."""
    if min == max:
        return ["{", str(min), "}"]
    if max == None:
        if min == 0:
            return ["*"]
        elif min == 1:
            return ["+"]
        else:
            return ["{", str(min), ",", "}"]
    if min == 0:
        if max == 1:
            return ["?"]
        else:
            return ["{", ",", str(max), "}"]
    return ["{", str(min), ",", str(max), "}"]


####################
# string matchers
####################


class Matcher(AstNode):
    pass


class HoleMatcher(Matcher):
    def __str__(self):
        return config.HOLE_GLYPH

    def __eq__(self, value):
        return isinstance(value, HoleMatcher)

    def is_hole(self):
        return True

    def num_matcher_holes(self):
        return 1


class ExactMatcher(Matcher):
    def __init__(self, s: Text):
        self.string = s

    def __str__(self):
        if is_identifier(self.string):
            # don't surround identifiers with quotes
            return self.string
        else:
            return json.dumps(self.string)

    def __eq__(self, value):
        return isinstance(value, ExactMatcher) and self.string == value.string


####################
# token constraints
####################


class Constraint(AstNode):
    pass


class HoleConstraint(Constraint):
    def __str__(self):
        return config.HOLE_GLYPH

    def __eq__(self, value):
        return isinstance(value, HoleConstraint)

    def is_hole(self):
        return True

    def num_constraint_holes(self):
        return 1

    def expand_leftmost_hole(self, vocabularies):
        return [
            FieldConstraint(HoleMatcher(), HoleMatcher()),
            NotConstraint(HoleConstraint()),
            AndConstraint(HoleConstraint(), HoleConstraint()),
            OrConstraint(HoleConstraint(), HoleConstraint()),
        ]


class FieldConstraint(Constraint):
    def __init__(self, name: Matcher, value: Matcher):
        self.name = name
        self.value = value

    def __str__(self):
        return f"{self.name}={self.value}"

    def __eq__(self, value):
        return (
            isinstance(value, FieldConstraint)
            and self.name == value.name
            and self.value == value.value
        )

    def has_holes(self):
        return self.name.has_holes() or self.value.has_holes()

    def tokens(self):
        return self.name.tokens() + ["="] + self.value.tokens()

    def num_matcher_holes(self):
        return self.name.num_matcher_holes() + self.value.num_matcher_holes()

    def expand_leftmost_hole(self, vocabularies):
        if self.name.is_hole():
            return [FieldConstraint(ExactMatcher(k), self.value) for k in vocabularies]
        elif self.value.is_hole():
            return [
                FieldConstraint(self.name, ExactMatcher(v))
                for v in vocabularies[self.name.string]
            ]
        else:
            return []

    def preorder_traversal(self):
        return (
            super().preorder_traversal()
            + self.name.preorder_traversal()
            + self.value.preorder_traversal()
        )


class NotConstraint(Constraint):
    def __init__(self, c: Constraint):
        self.constraint = c

    def __str__(self):
        c = maybe_parens(self.constraint, (AndConstraint, OrConstraint))
        return f"!{c}"

    def __eq__(self, value):
        return isinstance(value, NotConstraint) and self.constraint == value.constraint

    def has_holes(self):
        return self.constraint.has_holes()

    def tokens(self):
        return ["!"] + maybe_parens_tokens(
            self.constraint, (AndConstraint, OrConstraint)
        )

    def num_matcher_holes(self):
        return self.constraint.num_matcher_holes()

    def num_constraint_holes(self):
        return self.constraint.num_constraint_holes()

    def expand_leftmost_hole(self, vocabularies):
        # get the next nodes for the nested constraint
        nodes = self.constraint.expand_leftmost_hole(vocabularies)
        # avoid nesting negations
        return [NotConstraint(n) for n in nodes if not isinstance(n, NotConstraint)]

    def preorder_traversal(self):
        return super().preorder_traversal() + self.constraint.preorder_traversal()


class AndConstraint(Constraint):
    def __init__(self, lhs: Constraint, rhs: Constraint):
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return f"{self.lhs} & {self.rhs}"

    def __eq__(self, value):
        return (
            isinstance(value, AndConstraint)
            and self.lhs == value.lhs
            and self.rhs == value.rhs
        )

    def has_holes(self):
        return self.lhs.has_holes() or self.rhs.has_holes()

    def tokens(self):
        tokens = []
        tokens += maybe_parens_tokens(self.lhs, OrConstraint)
        tokens.append("&")
        tokens += maybe_parens_tokens(self.rhs, OrConstraint)
        return tokens

    def num_matcher_holes(self):
        return self.lhs.num_matcher_holes() + self.rhs.num_matcher_holes()

    def num_constraint_holes(self):
        return self.lhs.num_constraint_holes() + self.rhs.num_constraint_holes()

    def expand_leftmost_hole(self, vocabularies):
        if self.lhs.has_holes():
            nodes = self.lhs.expand_leftmost_hole(vocabularies)
            return [AndConstraint(n, self.rhs) for n in nodes]
        elif self.rhs.has_holes():
            nodes = self.rhs.expand_leftmost_hole(vocabularies)
            return [AndConstraint(self.lhs, n) for n in nodes]
        else:
            return []

    def preorder_traversal(self):
        return (
            super().preorder_traversal()
            + self.lhs.preorder_traversal()
            + self.rhs.preorder_traversal()
        )


class OrConstraint(Constraint):
    def __init__(self, lhs: Constraint, rhs: Constraint):
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return f"{self.lhs} | {self.rhs}"

    def __eq__(self, value):
        return (
            isinstance(value, OrConstraint)
            and self.lhs == value.lhs
            and self.rhs == value.rhs
        )

    def has_holes(self):
        return self.lhs.has_holes() or self.rhs.has_holes()

    def tokens(self):
        return [*self.lhs.tokens(), "|", *self.rhs.tokens()]

    def num_matcher_holes(self):
        return self.lhs.num_matcher_holes() + self.rhs.num_matcher_holes()

    def num_constraint_holes(self):
        return self.lhs.num_constraint_holes() + self.rhs.num_constraint_holes()

    def expand_leftmost_hole(self, vocabularies):
        if self.lhs.has_holes():
            nodes = self.lhs.expand_leftmost_hole(vocabularies)
            return [OrConstraint(n, self.rhs) for n in nodes]
        elif self.rhs.has_holes():
            nodes = self.rhs.expand_leftmost_hole(vocabularies)
            return [OrConstraint(self.lhs, n) for n in nodes]
        else:
            return []

    def preorder_traversal(self):
        return (
            super().preorder_traversal()
            + self.lhs.preorder_traversal()
            + self.rhs.preorder_traversal()
        )


####################
# surface patterns
####################


class Surface(AstNode):
    pass


class HoleSurface(Surface):
    def __str__(self):
        return config.HOLE_GLYPH

    def __eq__(self, value):
        return isinstance(value, HoleSurface)

    def is_hole(self):
        return True

    def num_surface_holes(self):
        return 1

    def expand_leftmost_hole(self, vocabularies):
        return [
            TokenSurface(HoleConstraint()),
            ConcatSurface(HoleSurface(), HoleSurface()),
            OrSurface(HoleSurface(), HoleSurface()),
            RepeatSurface(HoleSurface(), 0, 1),
            RepeatSurface(HoleSurface(), 0, None),
            RepeatSurface(HoleSurface(), 1, None),
        ]


class TokenSurface(Surface):
    def __init__(self, c: Constraint):
        self.constraint = c

    def __str__(self):
        return f"[{self.constraint}]"

    def __eq__(self, value):
        return isinstance(value, TokenSurface) and self.constraint == value.constraint

    def has_holes(self):
        return self.constraint.has_holes()

    def tokens(self):
        return ["[", *self.constraint.tokens(), "]"]

    def num_matcher_holes(self):
        return self.constraint.num_matcher_holes()

    def num_constraint_holes(self):
        return self.constraint.num_constraint_holes()

    def expand_leftmost_hole(self, vocabularies):
        nodes = self.constraint.expand_leftmost_hole(vocabularies)
        return [TokenSurface(n) for n in nodes]

    def preorder_traversal(self):
        return super().preorder_traversal() + self.constraint.preorder_traversal()


class ConcatSurface(Surface):
    def __init__(self, lhs: Surface, rhs: Surface):
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        lhs = maybe_parens(self.lhs, OrSurface)
        rhs = maybe_parens(self.rhs, OrSurface)
        return f"{lhs} {rhs}"

    def __eq__(self, value):
        return (
            isinstance(value, ConcatSurface)
            and self.lhs == value.lhs
            and self.rhs == value.rhs
        )

    def has_holes(self):
        return self.lhs.has_holes() or self.rhs.has_holes()

    def tokens(self):
        tokens = []
        tokens += maybe_parens_tokens(self.lhs, OrSurface)
        tokens += maybe_parens_tokens(self.rhs, OrSurface)
        return tokens

    def num_matcher_holes(self):
        return self.lhs.num_matcher_holes() + self.rhs.num_matcher_holes()

    def num_constraint_holes(self):
        return self.lhs.num_constraint_holes() + self.rhs.num_constraint_holes()

    def num_surface_holes(self):
        return self.lhs.num_surface_holes() + self.rhs.num_surface_holes()

    def expand_leftmost_hole(self, vocabularies):
        if self.lhs.has_holes():
            nodes = self.lhs.expand_leftmost_hole(vocabularies)
            return [ConcatSurface(n, self.rhs) for n in nodes]
        elif self.rhs.has_holes():
            nodes = self.rhs.expand_leftmost_hole(vocabularies)
            return [ConcatSurface(self.lhs, n) for n in nodes]
        else:
            return []

    def preorder_traversal(self):
        return (
            super().preorder_traversal()
            + self.lhs.preorder_traversal()
            + self.rhs.preorder_traversal()
        )


class OrSurface(Surface):
    def __init__(self, lhs: Surface, rhs: Surface):
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return f"{self.lhs} | {self.rhs}"

    def __eq__(self, value):
        return (
            isinstance(value, OrSurface)
            and self.lhs == value.lhs
            and self.rhs == value.rhs
        )

    def has_holes(self):
        return self.lhs.has_holes() or self.rhs.has_holes()

    def tokens(self):
        return [*self.lhs.tokens(), "|", *self.rhs.tokens()]

    def num_matcher_holes(self):
        return self.lhs.num_matcher_holes() + self.rhs.num_matcher_holes()

    def num_constraint_holes(self):
        return self.lhs.num_constraint_holes() + self.rhs.num_constraint_holes()

    def num_surface_holes(self):
        return self.lhs.num_surface_holes() + self.rhs.num_surface_holes()

    def expand_leftmost_hole(self, vocabularies):
        if self.lhs.has_holes():
            nodes = self.lhs.expand_leftmost_hole(vocabularies)
            return [OrSurface(n, self.rhs) for n in nodes]
        elif self.rhs.has_holes():
            nodes = self.rhs.expand_leftmost_hole(vocabularies)
            return [OrSurface(self.lhs, n) for n in nodes]
        else:
            return []

    def preorder_traversal(self):
        return (
            super().preorder_traversal()
            + self.lhs.preorder_traversal()
            + self.rhs.preorder_traversal()
        )


class RepeatSurface(Surface):
    def __init__(self, surf: Surface, min: int, max: Optional[int]):
        self.surf = surf
        self.min = min
        self.max = max

    def __str__(self):
        surf = maybe_parens(self.surf, (ConcatSurface, OrSurface))
        quant = make_quantifier(self.min, self.max)
        return f"{surf}{quant}"

    def __eq__(self, value):
        return (
            isinstance(value, RepeatSurface)
            and self.surf == value.surf
            and self.min == value.min
            and self.max == value.max
        )

    def has_holes(self):
        return self.surf.has_holes()

    def tokens(self):
        tokens = []
        tokens += maybe_parens_tokens(self.surf, (ConcatSurface, OrSurface))
        tokens += make_quantifier_tokens(self.min, self.max)
        return tokens

    def num_matcher_holes(self):
        return self.surf.num_matcher_holes()

    def num_constraint_holes(self):
        return self.surf.num_constraint_holes()

    def num_surface_holes(self):
        return self.surf.num_surface_holes()

    def expand_leftmost_hole(self, vocabularies):
        nodes = self.surf.expand_leftmost_hole(vocabularies)
        # avoid nesting repetitions
        nodes = [n for n in nodes if not isinstance(n, RepeatSurface)]
        return [RepeatSurface(n, self.min, self.max) for n in nodes]

    def preorder_traversal(self):
        return super().preorder_traversal() + self.surf.preorder_traversal()
