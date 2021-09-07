import random
from collections import defaultdict
from typing import Dict, Optional, List, Text, Type
from odinson.ruleutils.queryast import *
from odinson.ruleutils.queryparser import parse_odinson_query, parse_traversal
from odinson.ruleutils import config

# type alias
Vocabularies = Dict[Text, List[Text]]


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


def random_surface(vocabularies: Vocabularies, n_iters: int = 1, **kwargs) -> Surface:
    if 'allow_wildcards' in kwargs:
        kwargs['allow_surface_wildcards'] = kwargs['allow_wildcards']
    tree = random_tree(HoleSurface(), vocabularies, n_iters, **kwargs)
    # hack: pass tree through parser to make it right-heavy
    tree = parse_odinson_query(str(tree))
    return tree


def random_traversal(
    vocabularies: Vocabularies, n_iters: int = 1, **kwargs
) -> Traversal:
    if 'allow_wildcards' in kwargs:
        kwargs['allow_traversal_wildcards'] = kwargs['allow_wildcards']
    tree = random_tree(HoleTraversal(), vocabularies, n_iters, **kwargs)
    # hack: pass tree through parser to make it right-heavy
    tree = parse_traversal(str(tree))
    return tree


def random_hybrid(
    vocabularies: Vocabularies, n_iters: int = 1, **kwargs
) -> HybridQuery:
    return HybridQuery(
        random_surface(vocabularies, n_iters, **kwargs),
        random_traversal(vocabularies, n_iters, **kwargs),
        random_query(vocabularies, n_iters, **kwargs),
    )


def random_query(vocabularies: Vocabularies, n_iters: int = 1, **kwargs) -> AstNode:
    if random.random() < 0.5:
        return random_surface(vocabularies, n_iters, **kwargs)
    else:
        return random_hybrid(vocabularies, n_iters, **kwargs)


def random_tree(
    root: AstNode, vocabularies: Vocabularies, n_iters: int, **kwargs
) -> AstNode:
    tree = root
    # for a few iterations pick randomly from all candidates
    for i in range(n_iters):
        if not tree.has_holes():
            break
        candidates = tree.expand_leftmost_hole(vocabularies, **kwargs)
        tree = random.choice(candidates)
    # now we start to fill all remaining holes
    while tree.has_holes():
        query_holes = tree.num_query_holes()
        traversal_holes = tree.num_traversal_holes()
        surface_holes = tree.num_surface_holes()
        constraint_holes = tree.num_constraint_holes()
        matcher_holes = tree.num_matcher_holes()

        def is_improvement(c):
            qh = c.num_query_holes()
            if qh < query_holes:
                return True
            if qh > query_holes:
                return False
            th = c.num_traversal_holes()
            if th < traversal_holes:
                return True
            if th > traversal_holes:
                return False
            sh = c.num_surface_holes()
            if sh < surface_holes:
                return True
            if sh > surface_holes:
                return False
            ch = c.num_constraint_holes()
            return ch <= constraint_holes

        # discard candidates that don't improve the tree
        candidates = tree.expand_leftmost_hole(vocabularies, **kwargs)
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
        elif isinstance(n, (IncomingLabelTraversal, OutgoingLabelTraversal)):
            label = n.label.string
            vocabularies[config.SYNTAX_FIELD].add(label)
    return {k: list(v) for k, v in vocabularies.items()}
