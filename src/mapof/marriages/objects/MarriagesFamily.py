#!/usr/bin/env python

from mapof.core.objects.Family import Family
from mapof.marriages.objects.Marriages import Marriages
import copy

import mapof.marriages.cultures.mallows as mallows
from mapof.core.utils import *

class MarriagesFamily(Family):

    def __init__(self,
                 culture_id: str = None,
                 family_id='none',
                 params: dict = None,
                 size: int = 1,
                 label: str = "none",
                 color: str = "black",
                 alpha: float = 1.,
                 ms: int = 20,
                 show=True,
                 marker='o',
                 starting_from: int = 0,
                 path: dict = None,
                 single: bool = False,

                 num_agents: int = None):

        super().__init__(culture_id=culture_id,
                         family_id=family_id,
                         params=params,
                         size=size,
                         label=label,
                         color=color,
                         alpha=alpha,
                         ms=ms,
                         show=show,
                         marker=marker,
                         starting_from=starting_from,
                         path=path,
                         single=single)

        self.num_agents = num_agents

    def _get_params_for_paths(self, j, extremes=False):
        path = self.path

        variable = path['variable']

        if 'extremes' in path:
            extremes = path['extremes']

        params = {}
        if extremes:
            params[variable] = j / (self.size - 1)
        elif not extremes:
            params[variable] = (j + 1) / (self.size + 1)

        if 'scale' in path:
            params[variable] *= path['scale']

        if 'start' in path:
            params[variable] += path['start']
        else:
            path['start'] = 0.

        if 'step' in path:
            params[variable] = path['start'] + j * path['step']

        return params, variable

    def prepare_family(self,
                       experiment_id=None,
                       is_exported=True):

        instances = {}
        _keys = []

        for j in range(self.size):

            params = copy.deepcopy(self.params)

            path = self.path
            variable = None
            if path is not None and 'variable' in path:
                new_params, variable = self._get_params_for_paths(j)
                params = {**params, **new_params}

            if params is not None and 'norm-phi' in params:
                params['phi'] = mallows.phi_from_relphi(self.num_agents,
                                                        relphi=params['norm-phi'])

            instance_id = get_instance_id(self.single, self.family_id, j)

            instance = Marriages(experiment_id,
                                 instance_id,
                                 culture_id=self.culture_id,
                                 num_agents=self.num_agents,
                                 label=self.label,
                                 variable=variable,
                                 is_imported=False,
                                 **params)


            instance.prepare_instance(is_exported=is_exported)

            instances[instance_id] = instance
            _keys.append(instance_id)

        self.instance_ids = _keys

        return instances
