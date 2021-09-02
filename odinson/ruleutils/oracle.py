import random
from collections import defaultdict
from typing import Dict, Optional, List, Text, Type
from odinson.ruleutils.queryast import *

# type alias
Vocabularies = Dict[Text, List[Text]]


def path_from_root(
    target: AstNode, vocabularies: Optional[Vocabularies] = None
) -> List[AstNode]:
    """
    Returns the sequence of transitions from the root of the search tree
    to the specified AstNode.
    """
    root = HoleSurface()
    if vocabularies is None:
        # If no vocabularies were provided then construct
        # the minimal vocabularies required to reach the target.
        vocabularies = make_minimal_vocabularies(target)
    oracle = Oracle(root, target, vocabularies)
    return list(oracle.traversal())


def random_tree(vocabularies: Vocabularies, n_iters: int = 10) -> AstNode:
    # start with a single hole
    tree = HoleSurface()
    # for a few iterations pick randomly from all candidates
    for i in range(n_iters):
        candidates = tree.expand_leftmost_hole(vocabularies)
        tree = random.choice(candidates)
        if not tree.has_holes():
            break
    # now we start to fill all remaining holes
    while tree.has_holes():
        surf_holes = tree.num_surface_holes()
        const_holes = tree.num_constraint_holes()

        def is_improvement(c):
            sh = c.num_surface_holes()
            ch = c.num_constraint_holes()
            return sh < surf_holes or (sh == surf_holes and ch <= const_holes)

        # discard candidates that don't improve the tree
        candidates = tree.expand_leftmost_hole(vocabularies)
        candidates = [c for c in candidates if is_improvement(c)]
        # pick from good candidates only
        tree = random.choice(candidates)
    return tree


class Oracle:
    def __init__(self, src: AstNode, dst: AstNode, vocabularies: Vocabularies):
        self.src = src
        self.dst = dst
        self.vocabularies = vocabularies
        # find traversal corresponding to dst node
        self.dst_traversal = self.dst.preorder_traversal()

    def traversal(self):
        current = self.src
        while True:
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
    return {k: list(v) for k, v in vocabularies.items()}
