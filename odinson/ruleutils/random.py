import random
from odinson.ruleutils.queryast import *
from odinson.ruleutils.queryparser import *
from odinson.ruleutils.config import Vocabularies


def random_query(vocabularies: Vocabularies, n_iters: int = 1, **kwargs) -> AstNode:
    if random.random() < 0.5:
        return random_surface(vocabularies, n_iters, **kwargs)
    else:
        return random_hybrid(vocabularies, n_iters, **kwargs)


def random_hybrid(
    vocabularies: Vocabularies, n_iters: int = 1, **kwargs
) -> HybridQuery:
    return HybridQuery(
        random_surface(vocabularies, n_iters, **kwargs),
        random_traversal(vocabularies, n_iters, **kwargs),
        random_query(vocabularies, n_iters, **kwargs),
    )


def random_surface(vocabularies: Vocabularies, n_iters: int = 1, **kwargs) -> Surface:
    if "allow_wildcards" in kwargs:
        kwargs["allow_surface_wildcards"] = kwargs["allow_wildcards"]
    if "allow_mentions" in kwargs:
        kwargs["allow_surface_mentions"] = kwargs["allow_mentions"]
    if "allow_alternations" in kwargs:
        kwargs["allow_surface_alternations"] = kwargs["allow_alternations"]
    if "allow_concatenations" in kwargs:
        kwargs["allow_surface_concatenations"] = kwargs["allow_concatenations"]
    if "allow_repetitions" in kwargs:
        kwargs["allow_surface_repetitions"] = kwargs["allow_repetitions"]
    tree = random_tree(HoleSurface(), vocabularies, n_iters, **kwargs)
    # hack: pass tree through parser to make it right-heavy
    tree = parse_odinson_query(str(tree))
    return tree


def random_traversal(
    vocabularies: Vocabularies, n_iters: int = 1, **kwargs
) -> Traversal:
    if "allow_wildcards" in kwargs:
        kwargs["allow_traversal_wildcards"] = kwargs["allow_wildcards"]
    if "allow_alternations" in kwargs:
        kwargs["allow_traversal_alternations"] = kwargs["allow_alternations"]
    if "allow_concatenations" in kwargs:
        kwargs["allow_traversal_concatenations"] = kwargs["allow_concatenations"]
    if "allow_repetitions" in kwargs:
        kwargs["allow_traversal_repetitions"] = kwargs["allow_repetitions"]
    tree = random_tree(HoleTraversal(), vocabularies, n_iters, **kwargs)
    # hack: pass tree through parser to make it right-heavy
    tree = parse_traversal(str(tree))
    return tree


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
