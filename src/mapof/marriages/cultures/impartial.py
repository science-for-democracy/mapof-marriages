import numpy as np

from mapof.marriages.cultures.register import (
    register_marriages_independent_culture,
    register_marriages_dependent_culture,
)


@register_marriages_independent_culture('impartial')
def generate_ic_votes(num_agents: int = None, **_kwargs):
    """
    Generates the votes based on the Impartial Culture model.
    """
    return [list(np.random.permutation(num_agents)) for _ in range(num_agents)]


@register_marriages_independent_culture('identity')
def generate_id_votes(num_agents: int = None, **_kwargs):
    """
    Generates the votes based on the Identity model.
    """

    return [list(range(num_agents)) for _ in range(num_agents)]


@register_marriages_independent_culture('group_impartial')
def generate_group_ic_votes(num_agents: int = None, proportion: int = 0.5, **_kwargs):
    """
    Generates the votes based on the Group Impartial Culture model.
    """

    size_1 = int(proportion * num_agents)
    size_2 = int(num_agents - size_1)

    votes_1 = [list(np.random.permutation(size_1)) +
               list(np.random.permutation([j for j in range(size_1, num_agents)]))
               for _ in range(size_1)]

    votes_2 = [list(np.random.permutation([j for j in range(size_1, num_agents)])) +
               list(np.random.permutation(size_1))
               for _ in range(size_2)]

    votes = votes_1 + votes_2

    return votes


@register_marriages_independent_culture('symmetric')
def generate_symmetric_votes(num_agents: int = None, **_kwargs):
    """
        Generates the votes based on the Symmetric model.
    """

    num_rounds = num_agents - 1

    def next(agents):
        first = agents[0]
        last = agents[-1]
        middle = agents[1:-1]
        new_agents = [first, last]
        new_agents.extend(middle)
        return new_agents

    agents = [i for i in range(num_agents)]
    rounds = []

    for _ in range(num_rounds):
        pairs = []
        for i in range(num_agents // 2):
            agent_1 = agents[i]
            agent_2 = agents[num_agents - 1 - i]
            pairs.append([agent_1, agent_2])
        rounds.append(pairs)
        agents = next(agents)

    votes = np.zeros([num_agents, num_agents], dtype=int)

    for pos, partition in enumerate(rounds):
        for x, y in partition:
            votes[x][pos+1] = y
            votes[y][pos+1] = x

    for i in range(num_agents):
        votes[i][0] = i

    return votes

@register_marriages_dependent_culture('asymmetric')
def generate_asymmetric_votes(num_agents: int = None, **_kwargs):
    """
    Generates the votes based on the Asymmetric model.
    """
    votes = [list(range(num_agents)) for _ in range(num_agents)]
    votes_left = [_rotate(vote, shift+1) for shift, vote in enumerate(votes)]
    votes = [list(range(num_agents)) for _ in range(num_agents)]
    votes_right = [_rotate(vote, shift) for shift, vote in enumerate(votes)]
    return [votes_left, votes_right]


def _rotate(vector, shift):
    shift = shift % len(vector)
    return vector[shift:] + vector[:shift]
