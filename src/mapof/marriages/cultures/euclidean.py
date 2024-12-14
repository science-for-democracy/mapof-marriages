import numpy as np

from prefsampling.core.euclidean import EuclideanSpace
from prefsampling.core.euclidean import euclidean_space_to_sampler

from mapof.marriages.cultures.mallows import _mallows_votes

from mapof.marriages.cultures.register import (
    register_marriages_independent_culture,
    register_marriages_dependent_culture,
)

def weighted_l1(a1, a2, w):
    total = 0
    for i in range(len(a1)):
        total += abs(a1[i] - a2[i]) * w[i]
    return total

@register_marriages_dependent_culture('euclidean')
def generate_euclidean_votes(
        num_agents: int = None,
        num_dimensions: int = 2,
        space: str = None,
        **_kwargs
):
    """
        Generates the votes based on the Euclidean model.
    """

    if space is None:
        space = EuclideanSpace.UNIFORM_CUBE

    sampler, sampler_params = euclidean_space_to_sampler(space, num_dimensions)
    sampler_params['num_points'] = num_agents

    left = np.array(sampler(**sampler_params))
    right = np.array(sampler(**sampler_params))

    left_votes = np.zeros([num_agents, num_agents], dtype=int)
    right_votes = np.zeros([num_agents, num_agents], dtype=int)
    distances = np.zeros([num_agents, num_agents], dtype=float)

    for v in range(num_agents):
        for c in range(num_agents):
            left_votes[v][c] = c
            distances[v][c] = np.linalg.norm(left[v] - right[c])
        left_votes[v] = [x for _, x in sorted(zip(distances[v], left_votes[v]))]

    for v in range(num_agents):
        for c in range(num_agents):
            right_votes[v][c] = c
            distances[v][c] = np.linalg.norm(right[v] - left[c])
        right_votes[v] = [x for _, x in sorted(zip(distances[v], right_votes[v]))]

    return [left_votes, right_votes]

@register_marriages_dependent_culture('mallows_euclidean')
def generate_mallows_euclidean_votes(num_agents: int = None,
                                     num_dimensions: int = 2,
                                     space: str = None,
                                     phi=0.5,
                                     **kwargs):
    """
        Generates the votes based on the Mallows on top of the Euclidean model.
    """

    if space is None:
        space = EuclideanSpace.UNIFORM_CUBE

    sampler, sampler_params = euclidean_space_to_sampler(space, num_dimensions)
    sampler_params['num_points'] = num_agents

    left = np.array(sampler(**sampler_params))
    right = np.array(sampler(**sampler_params))

    left_votes = np.zeros([num_agents, num_agents], dtype=int)
    right_votes = np.zeros([num_agents, num_agents], dtype=int)
    distances = np.zeros([num_agents, num_agents], dtype=float)

    for v in range(num_agents):
        for c in range(num_agents):
            left_votes[v][c] = c
            distances[v][c] = np.linalg.norm(left[v] - right[c])
        left_votes[v] = [x for _, x in sorted(zip(distances[v], left_votes[v]))]

    for v in range(num_agents):
        for c in range(num_agents):
            right_votes[v][c] = c
            distances[v][c] = np.linalg.norm(right[v] - left[c])
        right_votes[v] = [x for _, x in sorted(zip(distances[v], right_votes[v]))]

    left_votes = _mallows_votes(left_votes, phi)
    right_votes = _mallows_votes(right_votes, phi)

    return [left_votes, right_votes]


@register_marriages_dependent_culture('reverse_euclidean')
def generate_reverse_euclidean_votes(
        num_agents: int = None,
        num_dimensions=2,
        space: str = None,
        phi=0.5,
        proportion: float = 0.5,
        **_kwargs
):
    """
        Generates the votes based on the Reverse Euclidean model.
    """
    if space is None:
        space = EuclideanSpace.UNIFORM_CUBE

    sampler, sampler_params = euclidean_space_to_sampler(space, num_dimensions)
    sampler_params['num_points'] = num_agents

    left = np.array(sampler(**sampler_params))
    right = np.array(sampler(**sampler_params))

    left_votes = np.zeros([num_agents, num_agents], dtype=int)
    right_votes = np.zeros([num_agents, num_agents], dtype=int)
    distances = np.zeros([num_agents, num_agents], dtype=float)

    for v in range(num_agents):
        for c in range(num_agents):
            left_votes[v][c] = c
            distances[v][c] = np.linalg.norm(left[v] - right[c])
        left_votes[v] = [x for _, x in sorted(zip(distances[v], left_votes[v]))]

    for v in range(num_agents):
        for c in range(num_agents):
            right_votes[v][c] = c
            distances[v][c] = np.linalg.norm(right[v] - left[c])
        right_votes[v] = [x for _, x in sorted(zip(distances[v], right_votes[v]))]

    p = proportion
    for i in range(int(num_agents * (1. - p))):
        tmp = list(left_votes[i])
        tmp.reverse()
        left_votes[i] = tmp

        tmp = list(right_votes[i])
        tmp.reverse()
        right_votes[i] = tmp

    return [left_votes, right_votes]

@register_marriages_dependent_culture('expectation')
def generate_expectation_votes(num_agents: int = None,
                               num_dimensions: int = 2,
                               space: str = None,
                               std=0.1,
                               phi=0.5,
                               **_kwargs):
    """
        Generates the votes based on the Expectation model.
    """

    if space is None:
        space = EuclideanSpace.UNIFORM_CUBE

    sampler, sampler_params = euclidean_space_to_sampler(space, num_dimensions)
    sampler_params['num_points'] = num_agents

    left_agents_reality = np.array(sampler(**sampler_params))
    left_agents_wishes = np.zeros([num_agents, 2])

    right_agents_reality = np.array(sampler(**sampler_params))
    right_agents_wishes = np.zeros([num_agents, 2])

    for v in range(num_agents):
        # while agents_wishes[v][0] <= 0 or agents_wishes[v][0] >= 1:
        left_agents_wishes[v][0] = np.random.normal(left_agents_reality[v][0], std)
        # while agents_wishes[v][1] <= 0 or agents_wishes[v][1] >= 1:
        left_agents_wishes[v][1] = np.random.normal(left_agents_reality[v][1], std)

        # while agents_wishes[v][0] <= 0 or agents_wishes[v][0] >= 1:
        right_agents_wishes[v][0] = np.random.normal(right_agents_reality[v][0], std)
        # while agents_wishes[v][1] <= 0 or agents_wishes[v][1] >= 1:
        right_agents_wishes[v][1] = np.random.normal(right_agents_reality[v][1], std)

    left_votes = np.zeros([num_agents, num_agents], dtype=int)
    right_votes = np.zeros([num_agents, num_agents], dtype=int)
    distances = np.zeros([num_agents, num_agents], dtype=float)

    for v in range(num_agents):
        for c in range(num_agents):
            left_votes[v][c] = c
            distances[v][c] = np.linalg.norm(right_agents_reality[c] - left_agents_wishes[v])
        left_votes[v] = [x for _, x in sorted(zip(distances[v], left_votes[v]))]

    for v in range(num_agents):
        for c in range(num_agents):
            right_votes[v][c] = c
            distances[v][c] = np.linalg.norm(left_agents_reality[c] - right_agents_wishes[v])
        right_votes[v] = [x for _, x in sorted(zip(distances[v], right_votes[v]))]

    return [left_votes, right_votes]

@register_marriages_dependent_culture('fame')
def generate_fame_votes(num_agents: int = None,
                        num_dimensions: int = 2,
                        space: str = None,
                        radius=0.1,
                        **_kwargs):
    """
    Generates the votes based on the Fame model.
    """

    if space is None:
        space = EuclideanSpace.UNIFORM_CUBE

    sampler, sampler_params = euclidean_space_to_sampler(space, num_dimensions)
    sampler_params['num_points'] = num_agents

    left = np.array(sampler(**sampler_params))
    right = np.array(sampler(**sampler_params))

    left_rays = np.array([np.random.uniform(0, radius) for _ in range(num_agents)])
    right_rays = np.array([np.random.uniform(0, radius) for _ in range(num_agents)])

    left_votes = np.zeros([num_agents, num_agents], dtype=int)
    right_votes = np.zeros([num_agents, num_agents], dtype=int)
    distances = np.zeros([num_agents, num_agents], dtype=float)

    for v in range(num_agents):
        for c in range(num_agents):
            left_votes[v][c] = c
            distances[v][c] = np.linalg.norm(left[v] - right[c])
            distances[v][c] = distances[v][c] - right_rays[c]
        left_votes[v] = [x for _, x in sorted(zip(distances[v], left_votes[v]))]

    for v in range(num_agents):
        for c in range(num_agents):
            right_votes[v][c] = c
            distances[v][c] = np.linalg.norm(right[v] - left[c])
            distances[v][c] = distances[v][c] - left_rays[c]
        right_votes[v] = [x for _, x in sorted(zip(distances[v], right_votes[v]))]

    return [left_votes, right_votes]

@register_marriages_dependent_culture('attributes')
def generate_attributes_votes(num_agents: int = None,
                              num_dimensions: int = 2,
                              space: str = None,
                              **_kwargs):
    """
    Generates the votes based on the Attributes model.
    """

    if space is None:
        space = EuclideanSpace.UNIFORM_CUBE

    sampler, sampler_params = euclidean_space_to_sampler(space, num_dimensions)
    sampler_params['num_points'] = num_agents


    left_agents_skills = np.array(sampler(**sampler_params))
    left_agents_weights = np.array(sampler(**sampler_params))
    right_agents_skills = np.array(sampler(**sampler_params))
    right_agents_weights = np.array(sampler(**sampler_params))

    votes = np.zeros([num_agents, num_agents], dtype=int)
    left_votes = np.zeros([num_agents, num_agents], dtype=int)
    right_votes = np.zeros([num_agents, num_agents], dtype=int)
    distances = np.zeros([num_agents, num_agents], dtype=float)
    ones = np.ones([num_dimensions], dtype=float)

    for v in range(num_agents):
        for c in range(num_agents):
            left_votes[v][c] = c
            if num_dimensions == 1:
                distances[v][c] = (1. - right_agents_skills[c]) * left_agents_weights[v]
            else:
                distances[v][c] = weighted_l1(ones, right_agents_skills[c], left_agents_weights[v])
        left_votes[v] = [x for _, x in sorted(zip(distances[v], left_votes[v]))]

    for v in range(num_agents):
        for c in range(num_agents):
            right_votes[v][c] = c
            if num_dimensions == 1:
                distances[v][c] = (1. - left_agents_skills[c]) * right_agents_weights[v]
            else:
                distances[v][c] = weighted_l1(ones, left_agents_skills[c], right_agents_weights[v])
        right_votes[v] = [x for _, x in sorted(zip(distances[v], right_votes[v]))]

    return [left_votes, right_votes]
