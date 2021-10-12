import random as rand
import numpy as np
import math


def distance(dim, x_1, x_2):
    """ compute distance between two points """

    if dim == 1:
        return abs(x_1[0] - x_2[0])

    output = 0.
    for i in range(dim):
        output += (x_1[i] - x_2[i]) ** 2
    return output ** 0.5


def generate_approval_euclidean_election(num_voters=None, num_candidates=None, params=None):

    # 'p' should be lower than 0.5

    alpha = 4
    beta = alpha/(params['p']) - alpha

    dim = params['dim']

    rankings = np.zeros([num_voters, num_candidates], dtype=int)
    distances = np.zeros([num_voters, num_candidates])
    votes = []

    if 'shift' in params:
        shift = np.array([params['shift']**2 for _ in range(dim)])
        voters = np.random.rand(num_voters, dim) + shift
        candidates = np.random.rand(num_candidates, dim)
    elif 'gauss' in params:
        voters = np.random.rand(num_voters, dim)
        params['gauss'] /= 2
        num_candidates_in_group_a = int(params['gauss'] * num_candidates)
        num_candidates_in_group_b = num_voters - num_candidates_in_group_a
        scale_group_a = params['gauss']
        scale_group_b = 1 - params['gauss']
        loc_a = [float(1/3) for _ in range(dim)]
        loc_b = [float(2/3) for _ in range(dim)]
        candidates_group_a = np.random.normal(loc=loc_a, scale=scale_group_a,
                                              size=(num_candidates_in_group_a, dim))
        candidates_group_b = np.random.normal(loc=loc_b, scale=scale_group_b,
                                              size=(num_candidates_in_group_b, dim))
        candidates = np.concatenate((candidates_group_a, candidates_group_b), axis=0)
    else:
        voters = []
        candidates = []
        print("We need params for euclidean model!")

    for v in range(num_voters):
        for c in range(num_candidates):
            rankings[v][c] = c
            distances[v][c] = distance(dim, voters[v], candidates[c])
        rankings[v] = [x for _, x in sorted(zip(distances[v], rankings[v]))]

    for v in range(num_voters):
        k = int(np.random.beta(alpha, beta) * num_candidates)
        votes.append(set(rankings[v][0:k]))

    return votes










####################################################################################################
####################################################################################################
####################################################################################################


# def generate_approval_1d_interval_elections(model=None, num_voters=None,
#                                             num_candidates=None):
#     """ helper function: generate simple approval 2d elections"""
#
#     threshold = 0.125
#     model = "1d_interval"
#
#     voters = [0 for _ in range(num_voters)]
#     candidates = [0 for _ in range(num_candidates)]
#
#     votes = [set() for _ in range(num_voters)]
#
#     for j in range(num_voters):
#         voters[j] = get_rand(model)
#     voters = sorted(voters)
#
#     for j in range(num_candidates):
#         candidates[j] = get_rand(model)
#     candidates = sorted(candidates)
#
#     for j in range(num_voters):
#         for k in range(num_candidates):
#             if distance(1, [voters[j]], [candidates[k]]) < threshold:
#                 votes[j].add(k)
#
#     return votes
#
#
# def generate_approval_2d_disc_elections(model=None, num_voters=None,
#                                           num_candidates=None):
#     """ helper function: generate simple approval 2d elections"""
#
#     threshold = 0.25
#     model = "2d_disc"
#
#     voters = [[0, 0] for _ in range(num_voters)]
#     candidates = [[0, 0] for _ in range(num_candidates)]
#
#     votes = [set() for _ in range(num_voters)]
#
#     for j in range(num_voters):
#         voters[j] = get_rand(model)
#     voters = sorted(voters)
#
#     for j in range(num_candidates):
#         candidates[j] = get_rand(model)
#     candidates = sorted(candidates)
#
#     for j in range(num_voters):
#         for k in range(num_candidates):
#             if distance(2, voters[j], candidates[k]) < threshold:
#                 votes[j].add(k)
#
#     return votes


def generate_1d_gaussian_party(model=None, num_voters=None,
                           num_candidates=None, params=None):
    if params is None:
        params = {}
    if 'num_winners' not in params:
        params['num_winners'] = 1

    voters = [[] for _ in range(num_voters)]
    candidates = [[] for _ in range(num_candidates)]

    votes = np.zeros([num_voters, num_candidates], dtype=int)
    distances = np.zeros([num_voters, num_candidates], dtype=float)

    for j in range(params['num_parties']):
        for w in range(params['num_winners']):
            _id = j*params['num_winners'] + w
            candidates[_id] = [rand.gauss(params['party'][j][0], params['var'])]

    _min = min(candidates)[0]
    _max = max(candidates)[0]

    shift = [rand.random()/2.-1/4.]
    for j in range(num_voters):
        voters[j] = [rand.random() * (_max - _min) + _min + shift[0]]

    for j in range(num_voters):
        for k in range(num_candidates):
            votes[j][k] = k
            distances[j][k] = distance(1, voters[j], candidates[k])

        votes[j] = [x for _, x in sorted(zip(distances[j], votes[j]))]

    return votes


def generate_2d_gaussian_party(model=None, num_voters=None,
                           num_candidates=None, params=None):
    if params is None:
        params = {}
    if 'num_winners' not in params:
        params['num_winners'] = 1

    voters = [[] for _ in range(num_voters)]
    candidates = [[] for _ in range(num_candidates)]

    votes = np.zeros([num_voters, num_candidates], dtype=int)
    distances = np.zeros([num_voters, num_candidates], dtype=float)

    for j in range(params['num_parties']):
        for w in range(params['num_winners']):
            _id = j*params['num_winners'] + w
            # print(_id)
            # print(rand.gauss(params['party'][j][1], params['var']))
            # print(candidates[_id])
            candidates[_id] = [rand.gauss(params['party'][j][0][0], params['var']),
                               rand.gauss(params['party'][j][0][1], params['var'])]

    def column(matrix, i):
        return [row[i] for row in matrix]

    x_min = min(column(candidates, 0))
    x_max = max(column(candidates, 0))
    y_min = min(column(candidates, 1))
    y_max = max(column(candidates, 1))

    shift = [rand.random()/2.-1/4., rand.random()/2.-1/4.]
    for j in range(num_voters):
        voters[j] = [rand.random() * (x_max - x_min) + x_min + shift[0],
                     rand.random() * (y_max - y_min) + y_min + shift[1]]


    # tmp_v = np.asarray(voters).transpose()
    # plt.scatter(tmp_v[0], tmp_v[1], color='grey')
    # tmp_c = np.asarray(candidates).transpose()
    # plt.scatter(tmp_c[0], tmp_c[1], color='blue')
    # plt.show()
    #
    # print(num_candidates, params['num_winners'], params['num_parties'])
    # print(candidates)


    for j in range(num_voters):
        for k in range(num_candidates):
            votes[j][k] = k
            distances[j][k] = distance(2, voters[j], candidates[k])

        votes[j] = [x for _, x in sorted(zip(distances[j], votes[j]))]

    return votes


def generate_elections_1d_simple(model=None, num_voters=None,
                                 num_candidates=None):
    """ helper function: generate simple 1d elections"""

    voters = [0 for _ in range(num_voters)]
    candidates = [0 for _ in range(num_candidates)]
    votes = np.zeros([num_voters, num_candidates], dtype=int)
    distances = np.zeros([num_voters, num_candidates], dtype=float)

    for j in range(num_voters):
        voters[j] = get_rand(model)
    voters = sorted(voters)

    for j in range(num_candidates):
        candidates[j] = get_rand(model)
    candidates = sorted(candidates)

    for j in range(num_voters):
        for k in range(num_candidates):
            votes[j][k] = k
            distances[j][k] = distance(1, voters[j], candidates[k])

        votes[j] = [x for _, x in sorted(zip(distances[j], votes[j]))]

    return votes


def generate_elections_2d_grid(model=None, num_voters=None,
                               num_candidates=None):

    voters = [[0, 0] for _ in range(num_voters)]
    candidates = []

    votes = np.zeros([num_voters, num_candidates], dtype=int)
    distances = np.zeros([num_voters, num_candidates], dtype=float)

    for j in range(num_voters):
        voters[j] = get_rand('2d_square')
    voters = sorted(voters)

    sq = int(num_candidates**0.5)
    d = 1./sq

    for i in range(sq):
        for j in range(sq):
            x = d/2. + d*i
            y = d/2. + d*j
            point = [x, y]
            candidates.append(point)

    for j in range(num_voters):
        for k in range(num_candidates):
            votes[j][k] = k
            distances[j][k] = distance(2, voters[j], candidates[k])

        votes[j] = [x for _, x in sorted(zip(distances[j], votes[j]))]

    return votes


def generate_elections_2d_simple(model=None, num_voters=None,
                                 num_candidates=None):
    """ helper function: generate simple 2d elections"""

    voters = [[0, 0] for _ in range(num_voters)]
    candidates = [[0, 0] for _ in range(num_candidates)]

    votes = np.zeros([num_voters, num_candidates], dtype=int)
    distances = np.zeros([num_voters, num_candidates], dtype=float)

    for j in range(num_voters):
        voters[j] = get_rand(model)
    voters = sorted(voters)

    for j in range(num_candidates):
        candidates[j] = get_rand(model)
    candidates = sorted(candidates)

    for j in range(num_voters):
        for k in range(num_candidates):
            votes[j][k] = k
            distances[j][k] = distance(2, voters[j], candidates[k])

        votes[j] = [x for _, x in sorted(zip(distances[j], votes[j]))]

    return votes


def generate_elections_nd_simple(model=None, num_voters=None,
                                 num_candidates=None):
    """ helper function: generate simple nd elections"""

    n_dim = 0

    if model == "3d_sphere" or model == "3d_cube" or \
            model == "3d_ball":
        n_dim = 3
    elif model == "4d_sphere" or model == "4d_cube" or \
            model == "4d_ball":
        n_dim = 4
    elif model == "5d_sphere" or model == "5d_cube" or \
            model == "5d_ball":
        n_dim = 5
    elif model == "10d_cube":
        n_dim = 10
    elif model == "15d_cube":
        n_dim = 15
    elif model == "20d_cube":
        n_dim = 20
    elif model == "40d_cube" or model == "40d_ball":
        n_dim = 40

    voters = [[0 for _ in range(n_dim)] for _ in range(num_voters)]
    candidates = [[0 for _ in range(n_dim)] for _ in range(num_candidates)]
    votes = np.zeros([num_voters, num_candidates], dtype=int)
    distances = np.zeros([num_voters, num_candidates], dtype=float)

    for j in range(num_voters):
        voters[j] = get_rand(model)
    voters = sorted(voters)

    for j in range(num_candidates):
        candidates[j] = get_rand(model)
    candidates = sorted(candidates)

    for j in range(num_voters):
        for k in range(num_candidates):
            votes[j][k] = k
            distances[j][k] = distance(n_dim, voters[j], candidates[k])

        votes[j] = [x for _, x in sorted(zip(distances[j], votes[j]))]

    return votes


# AUXILIARY
def random_ball(dimension, num_points=1, radius=1):
    from numpy import random, linalg
    # First generate random directions by normalizing the length of a
    # vector of random-normal values (these distribute evenly on ball).
    random_directions = random.normal(size=(dimension, num_points))
    random_directions /= linalg.norm(random_directions, axis=0)
    # Second generate a random radius with probability proportional to
    # the surface area of a ball with a given radius.
    random_radii = random.random(num_points) ** (1 / dimension)
    # Return the list of random (direction & length) points.
    return radius * (random_directions * random_radii).T


def get_rand(elections_model, cat="voters"):
    """ generate random values"""
    if elections_model in {"1d_interval", "1d_interval_bis"}:
        point = rand.random()
    elif elections_model in {"1d_gaussian", "1d_gaussian_bis"}:
        point = rand.gauss(0.5, 0.15)
        while point > 1 or point < 0:
            point = rand.gauss(0.5, 0.15)
    elif elections_model == "1d_one_sided_triangle":
        point = rand.uniform(0, 1) ** 0.5
    elif elections_model == "1d_full_triangle":
        point = rand.choice([rand.uniform(0, 1) ** 0.5,
                             2 - rand.uniform(0, 1) ** 0.5])
    elif elections_model == "1d_two_party":
        point = rand.choice([rand.uniform(0, 1), rand.uniform(2, 3)])
    elif elections_model in {"2d_disc", "2d_range_disc"}:
        phi = 2.0 * 180.0 * rand.random()
        radius = math.sqrt(rand.random()) * 0.5
        point = [0.5 + radius * math.cos(phi),
                 0.5 + radius * math.sin(phi)]
    elif elections_model == "2d_range_overlapping":
        phi = 2.0 * 180.0 * rand.random()
        radius = math.sqrt(rand.random()) * 0.5
        if cat == "voters":
            point = [0.25 + radius * math.cos(phi),
                     0.5 + radius * math.sin(phi)]
        elif cat == "candidates":
            point = [0.75 + radius * math.cos(phi),
                     0.5 + radius * math.sin(phi)]

    elif elections_model in {"2d_square"}:
        point = [rand.random(), rand.random()]

    elif elections_model == "2d_sphere":
        alpha = 2 * math.pi * rand.random()
        x = 1. * math.cos(alpha)
        y = 1. * math.sin(alpha)
        point = [x, y]

    elif elections_model in ["2d_gaussian", "2d_range_gaussian"]:
        point = [rand.gauss(0.5, 0.15), rand.gauss(0.5, 0.15)]
        while distance(2, point, [0.5, 0.5]) > 0.5:
            point = [rand.gauss(0.5, 0.15), rand.gauss(0.5, 0.15)]

    elif elections_model in ["2d_range_fourgau"]:
        r = rand.randint(1, 4)
        size = 0.06
        if r == 1:
            point = [rand.gauss(0.25, size), rand.gauss(0.5, size)]
        if r == 2:
            point = [rand.gauss(0.5, size), rand.gauss(0.75, size)]
        if r == 3:
            point = [rand.gauss(0.75, size), rand.gauss(0.5, size)]
        if r == 4:
            point = [rand.gauss(0.5, size), rand.gauss(0.25, size)]

    elif elections_model == "3d_interval_bis" or elections_model == "3d_cube":
        point = [rand.random(), rand.random(), rand.random()]
    elif elections_model == "3d_gaussian_bis":
        point = [rand.gauss(0.5, 0.15),
                 rand.gauss(0.5, 0.15),
                 rand.gauss(0.5, 0.15)]
        while distance(3, point, [0.5, 0.5, 0.5]) > 0.5:
            point = [rand.gauss(0.5, 0.15),
                     rand.gauss(0.5, 0.15),
                     rand.gauss(0.5, 0.15)]
    elif elections_model == "4d_cube":
        dim = 4
        point = [rand.random() for _ in range(dim)]
    elif elections_model == "5d_cube":
        dim = 5
        point = [rand.random() for _ in range(dim)]
    elif elections_model == "10d_cube":
        dim = 10
        point = [rand.random() for _ in range(dim)]
    elif elections_model == "20d_cube":
        dim = 20
        point = [rand.random() for _ in range(dim)]
    elif elections_model == "40d_cube":
        dim = 40
        point = [rand.random() for _ in range(dim)]
    elif elections_model == "3d_sphere":
        dim = 3
        point = list(random_ball(dim)[0])
    elif elections_model == "4d_sphere":
        dim = 4
        point = list(random_ball(dim)[0])
    elif elections_model == "5d_sphere":
        dim = 5
        point = list(random_ball(dim)[0])
    elif elections_model == "40d_ball":
        dim = 40
        point = list(random_ball(dim)[0])
    else:
        print('unknown model')
        point = [0, 0]
    return point


