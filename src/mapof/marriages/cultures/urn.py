import numpy as np

from mapof.marriages.cultures.register import register_marriages_independent_culture


@register_marriages_independent_culture('urn')
def generate_urn_votes(
        num_agents: int = None,
        alpha: float = 0,
        **_kwargs
):
    """
        Generates the votes based on the Urn model.
    """

    votes = np.zeros([num_agents, num_agents], dtype=int)
    urn_size = 1.
    for j in range(num_agents):
        rho = np.random.uniform(0, urn_size)
        if rho <= 1.:
            votes[j] = np.random.permutation(num_agents)
        else:
            votes[j] = votes[np.random.randint(0, j)]
        urn_size += alpha

    return votes