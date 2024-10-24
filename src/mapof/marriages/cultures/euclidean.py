import numpy as np
import math
from numpy import linalg
from mapof.marriages.cultures.mallows import mallows_votes


################################################################


def weighted_l1(a1, a2, w):
    total = 0
    for i in range(len(a1)):
        total += abs(a1[i] - a2[i]) * w[i]
    return total


def generate_euclidean_votes(num_agents: int = None,
                             dim=2,
                             space='uniform',
                             **kwargs):
    name = f'{dim}d_{space}'

    left = np.array([get_rand(name, i=i, num_agents=num_agents) for i in range(num_agents)])
    right = np.array([get_rand(name, i=i, num_agents=num_agents) for i in range(num_agents)])

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


def generate_mallows_euclidean_votes(num_agents: int = None,
                                     dim=2,
                                     space='uniform',
                                     phi=0.5,
                                     **kwargs):
    name = f'{dim}d_{space}'

    left = np.array([get_rand(name, i=i, num_agents=num_agents) for i in range(num_agents)])
    right = np.array([get_rand(name, i=i, num_agents=num_agents) for i in range(num_agents)])

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

    left_votes = mallows_votes(left_votes, phi)
    right_votes = mallows_votes(right_votes, phi)

    return [left_votes, right_votes]


def generate_reverse_euclidean_votes(num_agents: int = None,
                                     dim=2,
                                     space='uniform',
                                     phi=0.5,
                                    proportion: float = 0.5,
                                     **kwargs):
    name = f'{dim}d_{space}'

    left = np.array([get_rand(name, i=i, num_agents=num_agents) for i in range(num_agents)])
    right = np.array([get_rand(name, i=i, num_agents=num_agents) for i in range(num_agents)])

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


def generate_expectation_votes(num_agents: int = None,
                               dim=2,
                               space='uniform',
                               std=0.1,
                               phi=0.5,
                               **kwargs):
    name = f'{dim}d_{space}'

    left_agents_reality = np.array([get_rand(name) for _ in range(num_agents)])
    left_agents_wishes = np.zeros([num_agents, 2])

    right_agents_reality = np.array([get_rand(name) for _ in range(num_agents)])
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


def generate_fame_votes(num_agents: int = None,
                        dim=2,
                        space='uniform',
                        radius=0.1,
                        **kwargs):

    name = f'{dim}d_{space}'

    left = np.array([get_rand(name, i=i, num_agents=num_agents) for i in range(num_agents)])
    right = np.array([get_rand(name, i=i, num_agents=num_agents) for i in range(num_agents)])

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


def generate_attributes_votes(num_agents: int = None,
                              dim: int = 2,
                              space: str = 'uniform',
                              **kwargs):
    name = f'{dim}d_{space}'

    left_agents_skills = np.array([get_rand(name) for _ in range(num_agents)])
    left_agents_weights = np.array([get_rand(name) for _ in range(num_agents)])
    right_agents_skills = np.array([get_rand(name) for _ in range(num_agents)])
    right_agents_weights = np.array([get_rand(name) for _ in range(num_agents)])

    votes = np.zeros([num_agents, num_agents], dtype=int)
    left_votes = np.zeros([num_agents, num_agents], dtype=int)
    right_votes = np.zeros([num_agents, num_agents], dtype=int)
    distances = np.zeros([num_agents, num_agents], dtype=float)
    ones = np.ones([dim], dtype=float)

    for v in range(num_agents):
        for c in range(num_agents):
            left_votes[v][c] = c
            if dim == 1:
                distances[v][c] = (1. - right_agents_skills[c]) * left_agents_weights[v]
            else:
                distances[v][c] = weighted_l1(ones, right_agents_skills[c], left_agents_weights[v])
        left_votes[v] = [x for _, x in sorted(zip(distances[v], left_votes[v]))]

    for v in range(num_agents):
        for c in range(num_agents):
            right_votes[v][c] = c
            if dim == 1:
                distances[v][c] = (1. - left_agents_skills[c]) * right_agents_weights[v]
            else:
                distances[v][c] = weighted_l1(ones, left_agents_skills[c], right_agents_weights[v])
        right_votes[v] = [x for _, x in sorted(zip(distances[v], right_votes[v]))]

    return [left_votes, right_votes]


####################################################
### UPDATE THIS TO MATCH THE RESAMPLING APPROACH ###
def get_rand(model: str, i: int = 0, num_agents: int = 0) -> list:
    """ generate random values"""

    point = [0]
    if model in {"1d_uniform", "1d_interval"}:
        return np.random.rand()
    elif model in {'1d_asymmetric'}:
        if np.random.rand() < 0.3:
            return np.random.normal(loc=0.25, scale=0.15, size=1)
        else:
            return np.random.normal(loc=0.75, scale=0.15, size=1)
    elif model in {"1d_gaussian"}:
        point = np.random.normal(0.5, 0.15)
        while point > 1 or point < 0:
            point = np.random.normal(0.5, 0.15)
    # elif model == "1d_one_sided_triangle":
    #     point = np.random.uniform(0, 1) ** 0.5
    # elif model == "1d_full_triangle":
    #     point = np.random.choice(
    #         [np.random.uniform(0, 1) ** 0.5, 2 - np.random.uniform(0, 1) ** 0.5])
    # elif model == "1d_two_party":
    #     point = np.random.choice([np.random.uniform(0, 1), np.random.uniform(2, 3)])
    elif model in {"2d_disc"}:
        phi = 2.0 * 180.0 * np.random.random()
        radius = math.sqrt(np.random.random()) * 0.5
        point = [0.5 + radius * math.cos(phi), 0.5 + radius * math.sin(phi)]
    elif model in {"2d_square", "2d_uniform"}:
        point = [np.random.random(), np.random.random()]
    # elif model in {'2d_asymmetric'}:
    #     if np.random.rand() < 0.3:
    #         return np.random.normal(loc=0.25, scale=0.15, size=2)
    #     else:
    #         return np.random.normal(loc=0.75, scale=0.15, size=2)
    # elif model == "2d_sphere":
    #     alpha = 2 * math.pi * np.random.random()
    #     x = 1. * math.cos(alpha)
    #     y = 1. * math.sin(alpha)
    #     point = [x, y]
    # elif model in ["2d_gaussian"]:
    #     point = [np.random.normal(0.5, 0.15), np.random.normal(0.5, 0.15)]
    #     while np.linalg.norm(point - np.array([0.5, 0.5])) > 0.5:
    #         point = [np.random.normal(0.5, 0.15), np.random.normal(0.5, 0.15)]
    # elif model in ["3d_cube", "3d_uniform"]:
    #     point = [np.random.random(), np.random.random(), np.random.random()]
    # elif model in ["5d_uniform"]:
    #     dim = 5
    #     point = [np.random.random() for _ in range(dim)]
    # elif model in ["10d_uniform"]:
    #     dim = 10
    #     point = [np.random.random() for _ in range(dim)]
    # elif model in {'3d_asymmetric'}:
    #     if np.random.rand() < 0.3:
    #         return np.random.normal(loc=0.25, scale=0.15, size=3)
    #     else:
    #         return np.random.normal(loc=0.75, scale=0.15, size=3)
    # elif model in ['3d_gaussian']:
    #     point = [np.random.normal(0.5, 0.15),
    #              np.random.normal(0.5, 0.15),
    #              np.random.normal(0.5, 0.15)]
    #     while np.linalg.norm(point - np.array([0.5, 0.5, 0.5])) > 0.5:
    #         point = [np.random.normal(0.5, 0.15),
    #                  np.random.normal(0.5, 0.15),
    #                  np.random.normal(0.5, 0.15)]
    # elif model == "4d_cube":
    #     dim = 4
    #     point = [np.random.random() for _ in range(dim)]
    # elif model == "5d_cube":
    #     dim = 5
    #     point = [np.random.random() for _ in range(dim)]
    else:
        print('unknown culture_id', model)
        point = [0, 0]
    return point
