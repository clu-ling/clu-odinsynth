from collections import defaultdict
from typing import Dict, Optional, List
from odinson.ruleutils.queryast import *
from odinson.ruleutils.config import Vocabularies, ENTITY_FIELD, SYNTAX_FIELD


def make_transition_table(paths):
    """Gets a list of paths and returns a transition table."""
    trans = defaultdict(set)
    for path in paths:
        for i in range(len(path) - 1):
            trans[path[i]].add(path[i + 1])
    return {k: list(v) for k, v in trans.items()}


def all_paths_from_root(
    target: AstNode, vocabularies: Optional[Vocabularies] = None
) -> List[List[AstNode]]:
    """Returns all episodes that render an equivalent rule to target."""
    results = []
    for p in target.permutations():
        results.append(path_from_root(p, vocabularies))
    return results


def path_from_root(
    target: AstNode, vocabularies: Optional[Vocabularies] = None
) -> List[AstNode]:
    """
    Returns the sequence of transitions from the root of the search tree
    to the specified AstNode.
    """
    if isinstance(target, Query):
        root = HoleQuery()
    elif isinstance(target, Traversal):
        root = HoleTraversal()
    elif isinstance(target, Surface):
        root = HoleSurface()
    elif isinstance(target, Constraint):
        root = HoleConstraint()
    else:
        raise ValueError(f"unsupported target type '{type(target)}'")
    if vocabularies is None:
        # If no vocabularies were provided then construct
        # the minimal vocabularies required to reach the target.
        vocabularies = make_minimal_vocabularies(target)
    oracle = Oracle(root, target, vocabularies)
    return list(oracle.traversal())


class Oracle:
    def __init__(self, src: AstNode, dst: AstNode, vocabularies: Vocabularies):
        self.src = src
        self.dst = dst
        self.vocabularies = vocabularies
        # find traversal corresponding to dst node
        self.dst_traversal = self.dst.preorder_traversal()

    def traversal(self):
        current = self.src
        while current is not None:
            yield current
            if current == self.dst:
                break
            current = self.next_step(current)

    def next_step(self, current: AstNode):
        """Returns the next step in the path from src to dst."""
        # find position of first hole in current node's traversal
        hole_position = -1
        for i, n in enumerate(current.preorder_traversal()):
            if n.is_hole():
                hole_position = i
                break
        # if there is no hole then there is no next step
        if hole_position < 0:
            return
        # consider all possible candidates
        for candidate in current.expand_leftmost_hole(self.vocabularies):
            traversal = candidate.preorder_traversal()
            n1 = traversal[hole_position]
            n2 = self.dst_traversal[hole_position]
            if are_compatible(n1, n2):
                # if candidate has a node in the right position of its traversal
                # that is compatible with the node at the same position in the dst traversal
                # then we have a winner
                return candidate


def are_compatible(x: AstNode, y: AstNode) -> bool:
    """
    Compares two nodes to see if they're compatible.
    Note that this does not compare for equality,
    because the nodes may contain holes.
    """
    if isinstance(x, ExactMatcher) and isinstance(y, ExactMatcher):
        return x.string == y.string
    elif isinstance(x, RepeatSurface) and isinstance(y, RepeatSurface):
        return x.min == y.min and x.max == y.max
    elif isinstance(x, HoleSurface) and isinstance(y, Surface):
        return True
    else:
        return type(x) == type(y)


def make_minimal_vocabularies(node: AstNode) -> Vocabularies:
    """Returns the collection of vocabularies required to build the given rule."""
    vocabularies = defaultdict(set)
    for n in node.preorder_traversal():
        if isinstance(n, FieldConstraint):
            name = n.name.string
            value = n.value.string
            vocabularies[name].add(value)
        if isinstance(n, MentionSurface):
            label = n.label.string
            vocabularies[ENTITY_FIELD].add(label)
        elif isinstance(n, (IncomingLabelTraversal, OutgoingLabelTraversal)):
            label = n.label.string
            vocabularies[SYNTAX_FIELD].add(label)
    return {k: list(v) for k, v in vocabularies.items()}
