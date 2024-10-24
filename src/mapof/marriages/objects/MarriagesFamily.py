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

    def prepare_family(self,
                       experiment_id=None,
                       is_exported=True):

        instances = {}
        _keys = []

        for j in range(self.size):

            params = copy.deepcopy(self.params)

            variable = None
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
