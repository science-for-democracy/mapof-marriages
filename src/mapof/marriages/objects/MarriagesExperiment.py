import ast
import csv
import os
from abc import ABC

from mapof.core.objects.Experiment import Experiment
from mapof.core.persistence.experiment_imports import get_values_from_csv_file
from mapof.core.utils import make_folder_if_do_not_exist
from tqdm import tqdm

import mapof.marriages.features as features
from mapof.marriages.distances import get_distance
from mapof.marriages.objects.Marriages import Marriages
from mapof.marriages.objects.MarriagesFamily import MarriagesFamily

try:
    from sklearn.manifold import MDS
    from sklearn.manifold import TSNE
    from sklearn.manifold import SpectralEmbedding
    from sklearn.manifold import LocallyLinearEmbedding
    from sklearn.manifold import Isomap
except ImportError as error:
    MDS = None
    TSNE = None
    SpectralEmbedding = None
    LocallyLinearEmbedding = None
    Isomap = None
    print(error)


class MarriagesExperiment(Experiment, ABC):
    """Abstract set of elections."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.default_num_agents = 10

        self.matchings = {}

    def add_culture(self, name, function):
        pass

    def add_distance(self, name, function):
        pass

    def add_feature(self, name, function):
        pass

    def add_family(self,
                   culture_id: str = "none",
                   instance_id: str = None,
                   params: dict = None,
                   size: int = 1,
                   label: str = None,
                   color: str = "black",
                   alpha: float = 1.,
                   show: bool = True,
                   marker: str = 'o',
                   family_id: str = None,
                   single_instance: bool = False,
                   num_agents: int = None
                   ):

        if instance_id is not None:
            family_id = instance_id

        if num_agents is None:
            num_agents = self.default_num_agents

        if self.families is None:
            self.families = {}

        if params is None:
            params = {}

        if family_id is None:
            family_id = culture_id + '_' + str(num_agents)

        if label is None:
            label = family_id

        self.families[family_id] = MarriagesFamily(
            culture_id=culture_id,
            family_id=family_id,
            params=params,
            label=label,
            color=color,
            alpha=alpha,
            single=single_instance,
            show=show,
            size=size,
            marker=marker,
            num_agents=num_agents,
        )

        self.num_families = len(self.families)
        self.num_instances = sum([self.families[family_id].size for family_id in self.families])

        new_instances = self.families[family_id].prepare_family(
            is_exported=self.is_exported,
            experiment_id=self.experiment_id)

        for instance_id in new_instances:
            self.instances[instance_id] = new_instances[instance_id]

        self.families[family_id].instance_ids = list(new_instances.keys())

        # if self.is_exported:
        #     self.update_map_csv()  # To be implemented

        return list(new_instances.keys())

    def add_instance(self,
                     culture_id="none",
                     params=None,
                     label=None,
                     color="black",
                     alpha=1.,
                     show=True,
                     marker='x',
                     starting_from=0,
                     size=1,
                     num_agents=None,
                     instance_id=None):

        if num_agents is None:
            num_agents = self.default_num_agents

        return self.add_family(
            culture_id=culture_id,
            instance_id=instance_id,
            params=params,
            size=size,
            label=label,
            color=color,
            alpha=alpha,
            show=show,
            marker=marker,
            family_id=instance_id,
            num_agents=num_agents,
            single_instance=True,
        )

    def add_instances_to_experiment(self):

        instances = {}

        for family_id in self.families:

            ids = []
            if self.families[family_id].single:
                instance_id = family_id
                instance = Marriages(self.experiment_id, instance_id)
                instances[instance_id] = instance
                ids.append(str(instance_id))
            else:
                for j in range(self.families[family_id].size):
                    instance_id = family_id + '_' + str(j)
                    instance = Marriages(self.experiment_id, instance_id)
                    instances[instance_id] = instance
                    ids.append(str(instance_id))

            self.families[family_id].instance_ids = ids

        return instances

    def get_distance(self,
                     election_1,
                     election_2,
                     distance_id: str = None,
                     **kwargs
                     ) -> float or (float, list):
        return get_distance(election_1, election_2, distance_id)

    def import_controllers(self):
        """ Import controllers from a file """

        families = {}

        path = os.path.join(os.getcwd(), 'experiments', self.experiment_id, 'map.csv')
        file_ = open(path, 'r')

        header = [h.strip() for h in file_.readline().split(';')]
        reader = csv.DictReader(file_, fieldnames=header, delimiter=';')

        starting_from = 0
        for row in reader:

            culture_id = None
            color = None
            label = None
            params = None
            alpha = None
            size = None
            marker = None
            num_agents = None
            family_id = None
            show = True

            if 'culture_id' in row.keys():
                culture_id = str(row['culture_id']).strip()

            if 'color' in row.keys():
                color = str(row['color']).strip()

            if 'label' in row.keys():
                label = str(row['label'])

            if 'family_id' in row.keys():
                family_id = str(row['family_id'])

            if 'params' in row.keys():
                params = ast.literal_eval(str(row['params']))

            if 'alpha' in row.keys():
                alpha = float(row['alpha'])

            if 'size' in row.keys():
                size = int(row['size'])

            if 'marker' in row.keys():
                marker = str(row['marker']).strip()

            if 'num_agents' in row.keys():
                num_agents = int(row['num_agents'])

            if 'show' in row.keys():
                show = row['show'].strip() == 'process_id'

            single_instance = size == 1

            families[family_id] = MarriagesFamily(culture_id=culture_id,
                                                  family_id=family_id,
                                                  params=params, label=label,
                                                  color=color, alpha=alpha, show=show,
                                                  size=size, marker=marker,
                                                  starting_from=starting_from,
                                                  num_agents=num_agents,
                                                  single=single_instance)
            starting_from += size

        self.num_families = len(families)
        self.num_instances = sum([families[family_id].size for family_id in families])
        self.main_order = [i for i in range(self.num_instances)]

        file_.close()
        return families

    def add_folders_to_experiment(self) -> None:

        dirs = ["experiments"]
        for dir in dirs:
            if not os.path.isdir(dir):
                os.mkdir(os.path.join(os.getcwd(), dir))

        if not os.path.isdir(os.path.join(os.getcwd(), "experiments", self.experiment_id)):
            os.mkdir(os.path.join(os.getcwd(), "experiments", self.experiment_id))

        list_of_folders = ['distances',
                           'features',
                           'coordinates',
                           'instances']

        for folder_name in list_of_folders:
            if not os.path.isdir(os.path.join(os.getcwd(), "experiments",
                                              self.experiment_id, folder_name)):
                os.mkdir(os.path.join(os.getcwd(), "experiments",
                                      self.experiment_id, folder_name))

        path = os.path.join(os.getcwd(), "experiments", self.experiment_id, "map.csv")
        if not os.path.exists(path):

            with open(path, 'w') as file_csv:
                file_csv.write(
                    "size;num_agents;culture_id;params;color;alpha;"
                    "family_id;label;marker\n")
                file_csv.write("3;10;ic;{};black;1;ic;Impartial Culture;o\n")

    def prepare_instances(self):

        if self.instances is None:
            self.instances = {}

        for family_id in tqdm(self.families, desc="Preparing instances"):

            new_instances = self.families[family_id].prepare_family(
                is_exported=self.is_exported,
                experiment_id=self.experiment_id)

            for instance_id in new_instances:
                self.instances[instance_id] = new_instances[instance_id]

    def compute_feature(self, feature_id: str = None, feature_params=None) -> dict:

        if feature_params is None:
            feature_params = {}

        feature_dict = {'value': {}, 'time': {}, 'std': {}}

        features_with_time = {}
        features_with_std = {'avg_num_of_bps_for_rand_matching',
                             'avg_number_of_bps_for_random_matching'}

        if feature_id == 'summed_rank_difference':
            minimal = get_values_from_csv_file(self, feature_id='summed_rank_minimal_matching')
            maximal = get_values_from_csv_file(self, feature_id='summed_rank_maximal_matching')

            for instance_id in self.instances:
                if minimal[instance_id] is None:
                    value = 'None'
                else:
                    value = abs(maximal[instance_id] - minimal[instance_id])
                feature_dict['value'][instance_id] = value
                feature_dict['time'][instance_id] = 0

        else:

            for instance_id in self.instances:
                feature = features.get_feature(feature_id)
                instance = self.instances[instance_id]
                value = feature(instance.votes)

                if feature_id in features_with_time:
                    feature_dict['value'][instance_id] = value[0]
                    feature_dict['time'][instance_id] = value[1]
                elif feature_id in features_with_std:
                    feature_dict['value'][instance_id] = value[0]
                    feature_dict['std'][instance_id] = value[1]
                else:
                    feature_dict['value'][instance_id] = value

        if self.is_exported:

            path_to_folder = os.path.join(os.getcwd(), "experiments", self.experiment_id,
                                          "features")
            make_folder_if_do_not_exist(path_to_folder)
            path_to_file = os.path.join(path_to_folder, f'{feature_id}.csv')

            with open(path_to_file, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file, delimiter=';')

                if feature_id in features_with_time:
                    writer.writerow(["instance_id", "value", 'time'])
                    for key in feature_dict['value']:
                        writer.writerow(
                            [key, feature_dict['value'][key], round(feature_dict['time'][key], 3)])
                elif feature_id in features_with_std:
                    writer.writerow(["instance_id", "value", 'std'])
                    for key in feature_dict['value']:
                        writer.writerow(
                            [key, feature_dict['value'][key], round(feature_dict['std'][key], 3)])
                else:
                    writer.writerow(["instance_id", "value"])
                    for key in feature_dict['value']:
                        writer.writerow([key, feature_dict['value'][key]])

        self.features[feature_id] = feature_dict
        return feature_dict

