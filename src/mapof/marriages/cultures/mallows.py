import mapof.core.features.mallows as ml

from mapof.marriages.cultures.impartial import generate_asymmetric_votes

from mapof.marriages.cultures.register import (
    register_marriages_independent_culture,
    register_marriages_dependent_culture,
)


@register_marriages_independent_culture('mallows')
def generate_mallows_votes(*args, **kwargs):
    return ml.generate_mallows_votes(*args, **kwargs)

@register_marriages_independent_culture('norm_mallows')
def generate_norm_mallows_votes(num_agents=None,
                                normphi=0.5,
                                weight=0.,
                                **kwargs):
    """
        Generates the votes based on the Norm-Mallows model.
    """

    phi = ml.phi_from_normphi(num_agents, normphi=normphi)

    return generate_mallows_votes(num_agents, num_agents, phi)


@register_marriages_dependent_culture('malasym')
def generate_mallows_asymmetric_votes(num_agents: int = None,
                                      phi: float = 0.5,
                                      **kwargs):
    """
        Generates the votes based on the Mallows model on top of Asymmetric instance.
    """

    votes_left, votes_right = generate_asymmetric_votes(num_agents=num_agents)

    votes_left = _mallows_votes(votes_left, phi)
    votes_right = _mallows_votes(votes_right, phi)

    return [votes_left, votes_right]


def _mallows_vote(vote, phi):
    num_candidates = len(vote)
    raw_vote = generate_mallows_votes(1, num_candidates, phi=phi, weight=0)[0]
    new_vote = [0] * len(vote)
    for i in range(num_candidates):
        new_vote[raw_vote[i]] = vote[i]
    return new_vote


def _mallows_votes(votes, phi):
    for i in range(len(votes)):
        votes[i] = _mallows_vote(votes[i], phi)
    return votes
