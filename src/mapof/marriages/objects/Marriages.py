from mapof.core.objects.Instance import Instance

import mapof.marriages.persistence.instance_exports as exports
from mapof.marriages.cultures import *
from mapof.marriages.features import get_feature
from mapof.marriages.persistence.instance_imports import import_real_instance


class Marriages(Instance):

    def __init__(self,
                 experiment_id,
                 instance_id,
                 culture_id=None,
                 num_agents=None,
                 is_imported=True,
                 votes=None,
                 params=None,
                 **kwargs):

        super().__init__(
            experiment_id,
            instance_id,
            culture_id=culture_id,
            params=params,
            **kwargs
        )

        self.num_agents = num_agents
        self.votes = votes

        self.retrospetive_vectors = None

        if is_imported and experiment_id != 'virtual':
            try:
                self.votes, self.num_agents, self.params, self.culture_id = import_real_instance(self)
            except Exception:
                logging.warning(f'Could not import instance {self.instance_id}.')

    def get_retrospective_vectors(self):
        if self.retrospetive_vectors is not None:
            return self.retrospetive_vectors
        else:
            return self.votes_to_retrospective_vectors()

    def votes_to_retrospective_vectors(self):

        vectors = np.zeros([2, self.num_agents, self.num_agents], dtype=int)

        for a in range(self.num_agents):
            for i, b in enumerate(self.votes[0][a]):
                vectors[0][a][i] = int(list(self.votes[1][b]).index(a))

        for a in range(self.num_agents):
            for i, b in enumerate(self.votes[1][a]):
                vectors[1][a][i] = int(list(self.votes[0][b]).index(a))

        self.retrospetive_vectors = vectors
        return vectors

    def prepare_instance(self, is_exported=None):

        if 'norm-phi' in self.params:  # for backward compatibility
            self.params['alpha'] = self.params['norm-phi']
        else:
            self.params['alpha'] = 1

        self.votes = generate_votes(culture_id=self.culture_id,
                                    num_agents=self.num_agents,
                                    params=self.params)

        if is_exported:
            exports.export_instance_to_a_file(self)

    def compute_feature(self, feature_id, feature_long_id=None, **kwargs):
        if feature_long_id is None:
            feature_long_id = feature_id
        feature = get_feature(feature_id)
        self.features[feature_long_id] = feature(self, **kwargs)
