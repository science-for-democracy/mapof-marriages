import statistics
import sys
from random import shuffle

import gurobipy as gp
import networkx as nx
from gurobipy import GRB

from mapof.marriages.features.register import register_marriages_feature

sys.setrecursionlimit(10000)
# warnings.filterwarnings("error")


def number_blocking_pairs(instance, matching) -> int:
    bps = 0
    num_agents = len(instance[0])
    for i in range(num_agents):
        for j in range(num_agents):
            partner_i = matching[0][i]
            partneri_index = instance[0][i].index(partner_i)
            partner_j = matching[1][j]
            partnerj_index = instance[1][j].index(partner_j)
            if instance[0][i].index(j) < partneri_index:
                if instance[1][j].index(i) < partnerj_index:
                    bps += 1
    return bps


def _rank_matching(instance, best, summed):
    """ # Only call for instances that admit a stable matchig
# summed: Set to true we try to optimize the summed rank of agents for their partner in the matching, set to false we optimize the minimum rank
# Best (only relevant if summed is set to true): Set to true we output the best possible matching, set to false the worst one
"""
    num_agents = len(instance[0])
    m = gp.Model("mip1")
    m.setParam('OutputFlag', False)
    x = m.addVars(num_agents, num_agents, lb=0, ub=1, vtype=GRB.BINARY)
    opt = m.addVar(vtype=GRB.INTEGER, lb=0, ub=num_agents * num_agents)
    opt1 = m.addVar(vtype=GRB.INTEGER, lb=0, ub=num_agents * num_agents)
    opt2 = m.addVar(vtype=GRB.INTEGER, lb=0, ub=num_agents * num_agents)
    for i in range(num_agents):
        m.addConstr(gp.quicksum(x[i, j] for j in range(num_agents)) <= 1)
        m.addConstr(gp.quicksum(x[j, i] for j in range(num_agents)) <= 1)
    for i in range(num_agents):
        for j in range(num_agents):
            better_pairs = []
            for t in range(0, instance[0][i].index(j) + 1):
                better_pairs.append([i, instance[0][i][t]])
            for t in range(0, instance[1][j].index(i) + 1):
                better_pairs.append([instance[1][j][t], j])
            m.addConstr(gp.quicksum(x[a[0], a[1]] for a in better_pairs) >= 1)
    if summed:
        m.addConstr(gp.quicksum(
            instance[0][i].index(j) * x[i, j] for i in range(num_agents) for j in
            range(num_agents)) == opt1)
        m.addConstr(gp.quicksum(
            instance[1][i].index(j) * x[j, i] for i in range(num_agents) for j in
            range(num_agents)) == opt2)
        m.addConstr(opt1 + opt2 == opt)
        if best:
            m.setObjective(opt, GRB.MAXIMIZE)
        else:
            m.setObjective(opt, GRB.MINIMIZE)
    else:
        for i in range(num_agents):
            m.addConstr(
                gp.quicksum(instance[0][j].index(i) * x[j, i] for j in range(num_agents)) <= opt)
            m.addConstr(
                gp.quicksum(instance[1][j].index(i) * x[i, j] for j in range(num_agents)) <= opt)
        m.setObjective(opt, GRB.MINIMIZE)
    m.optimize()
    matching1 = {}
    matching2 = {}
    for i in range(num_agents):
        for j in range(num_agents):
            if x[i, j].X == 1:
                matching1[i] = j
                matching2[j] = i
    return int(m.objVal), [matching1, matching2]

@register_marriages_feature('summed_rank_maximal_matching')
def summed_rank_maximal_matching(instance):
    try:
        val, matching = _rank_matching(instance, True, True)
    except:
        return None
    return val

@register_marriages_feature('summed_rank_minimal_matching')
def summed_rank_minimal_matching(instance):
    try:
        val, matching = _rank_matching(instance, False, True)
    except:
        return None

    return val

@register_marriages_feature('minimal_rank_maximizing_matching')
def minimal_rank_maximizing_matching(instance):
    try:
        val, matching = _rank_matching(instance, True, False)
    except Exception:
        return None
    return val


@register_marriages_feature('avg_num_of_bps_for_rand_matching')
def avg_number_of_bps_for_rand_matching(instance, iterations=100):
    instance = instance.votes
    bps = []
    num_agents = len(instance[0])
    for _ in range(iterations):
        agents = list(range(num_agents))
        shuffle(agents)
        m2 = []
        for i in range(num_agents):
            m2.append(agents.index(i))
        bps.append(number_blocking_pairs(instance, [agents, m2]))
    return statistics.mean(bps), statistics.stdev(bps)


@register_marriages_feature('num_of_bps_min_weight')
def number_of_bps_maximum_weight(instance):
    instance = instance.votes
    num_agents = len(instance[0])
    G = nx.Graph()
    for i in range(num_agents):
        for j in range(num_agents):
            G.add_edge(i, j + num_agents,
                       weight=2 * (num_agents - 1) - instance[0][i].index(j) - instance[1][j].index(
                           i))
    matching = nx.max_weight_matching(G, maxcardinality=True)
    matching_dict_m = {}
    matching_dict_w = {}
    for p in matching:
        small = min(p[0], p[1])
        big = max(p[0], p[1])
        matching_dict_m[small] = big - num_agents
        matching_dict_w[big - num_agents] = small
    return number_blocking_pairs(instance, [matching_dict_m, matching_dict_w])
