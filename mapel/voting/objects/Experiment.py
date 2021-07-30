#!/usr/bin/env python


from mapel.voting.objects.Election import Election
from mapel.voting.objects.Family import Family

import mapel.voting._elections as _elections
import mapel.voting.features as features

from threading import Thread
from time import sleep

import mapel.voting._metrics as metr

import mapel.voting.print as pr

import mapel.voting.elections.preflib as preflib

import math
import csv
import os

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


try:
    from sklearn.manifold import MDS
    from sklearn.manifold import TSNE
except:
    pass

from mapel.voting.glossary import NICE_NAME, LIST_OF_FAKE_MODELS



class Experiment:
    """Abstract set of elections."""

    def __init__(self, ignore=None, elections=None, distances=None, with_matrices=False, coordinates=None,
                 distance_name='emd-positionwise', import_controllers=True, experiment_id=None):

        self.distance_name = distance_name
        self.elections = {}
        self.default_num_candidate = 10
        self.default_num_voters = 100

        self.families = None
        self.distances = None
        self.times = None
        self.points_by_families = None

        if experiment_id is None:
            self.store = False
        else:
            self.store = True
            self.experiment_id = experiment_id
            self.create_structure()
            self.families = self.import_controllers(ignore=ignore)
            self.attraction_factor = 1

        if elections is not None:
            if elections == 'import':
                self.elections = self.add_elections_to_experiment(with_matrices=with_matrices)
            else:
                self.elections = elections

        if distances is not None:
            if distances == 'import':
                self.distances = self.add_distances_to_experiment()
            else:
                self.distances = distances

        if coordinates is not None:
            if coordinates == 'import':
                self.coordinates = self.add_coordinates_to_experiment()
            else:
                self.coordinates = coordinates

        self.features = {}

    def set_default_num_candidates(self, num_candidates):
        self.default_num_candidates = num_candidates

    def set_default_num_voters(self, num_voters):
        self.default_num_voters = num_voters

    def add_election(self, election_model="none", param_1=0., param_2=0., label=None,
                     color="black", alpha=1., show=True, marker='x', starting_from=0,
                     num_candidates=None, num_voters=None, election_id=None):

        if num_candidates is None:
            num_candidates = self.default_num_candidate

        if num_voters is None:
            num_voters = self.default_num_voters

        return self.add_family(election_model=election_model,
                        param_1=param_1, param_2=param_2,
                        size=1,
                        label=label,
                        color=color,
                        alpha=alpha, show=show,
                        marker=marker,
                        starting_from=starting_from,
                        num_candidates=num_candidates,
                        num_voters=num_voters,
                        family_id=election_id,
                        single_election=True)[0]

    def add_family(self, election_model="none", param_1=0., param_2=0., size=1, label=None,
                   color="black", alpha=1., show=True, marker='o', starting_from=0,
                   num_candidates=None, num_voters=None, family_id=None, single_election=False):
        """ Only add the Family; the Elections will be generated later"""

        if num_candidates is None:
            num_candidates = self.default_num_candidate

        if num_voters is None:
            num_voters = self.default_num_voters

        if self.families is None:
            self.families = {}

        if family_id is None:
            family_id = election_model + '_' + str(num_candidates) + '_' + str(num_voters)
            if election_model in {'urn_model', 'norm-mallows', 'mallows', 'norm-mallows_matrix'} and param_1 != 0:
                family_id += '_' + str(float(param_1))
            if election_model in {'norm-mallows', 'mallows'} and param_2 != 0:
                family_id += '__' + str(float(param_2))

        if label is None:
            label = family_id

        self.families[family_id] = Family(election_model=election_model, family_id=family_id,
                                          param_1=param_1, param_2=param_2, label=label,
                                          color=color, alpha=alpha, show=show, size=size, marker=marker,
                                          starting_from=starting_from,
                                          num_candidates=num_candidates, num_voters=num_voters,
                                          single_election=single_election)

        self.num_families = len(self.families)
        self.num_elections = sum([self.families[family_id].size for family_id in self.families])
        self.main_order = [i for i in range(self.num_elections)]

        ids = self.generate_family_elections(family_id)
        self.families[family_id].election_ids = ids

        return ids


    def generate_family_elections(self, family_id):
        param_1 = self.families[family_id].param_1
        param_2 = self.families[family_id].param_2
        # num_candidates = self.families[family_id].num_candidates
        # num_voters = self.families[family_id].num_voters

        election_model = self.families[family_id].election_model

        if election_model in preflib.LIST_OF_PREFLIB_MODELS:
            _elections.prepare_preflib_family(experiment=self, election_model=election_model,
                                              param_1=param_1)
        else:
           return _elections.prepare_statistical_culture_family(experiment=self,
                                                          election_model=election_model,
                                                          family_id=family_id,
                                                          param_1=param_1, param_2=param_2)

    def prepare_elections(self):
        """ Prepare elections for a given experiment """

        if self.elections is None:
            self.elections = {}

        if self.store:
            path = os.path.join(os.getcwd(), "experiments", self.experiment_id, "elections")
            for file_name in os.listdir(path):
                os.remove(os.path.join(path, file_name))

            for family_id in self.families:
                param_1 = self.families[family_id].param_1
                param_2 = self.families[family_id].param_2
                # num_candidates = self.families[family_id].num_candidates
                # num_voters = self.families[family_id].num_voters

                election_model = self.families[family_id].election_model

                if election_model in preflib.LIST_OF_PREFLIB_MODELS:
                    _elections.prepare_preflib_family(experiment=self, election_model=election_model,
                                                      param_1=param_1)
                else:
                    _elections.prepare_statistical_culture_family(experiment=self,
                                                                  election_model=election_model,
                                                                  family_id=family_id,
                                                                  param_1=param_1, param_2=param_2)

    def compute_distances(self, distance_name='emd-positionwise', num_threads=1, self_distances=False):
        """ Compute distances between elections (using threads)"""

        self.distance_name=distance_name

        distances = {}
        times = {}
        for election_id in self.elections:
            distances[election_id] = {}
            times[election_id] = {}

        threads = [{} for _ in range(num_threads)]

        ids = []
        for i, election_1 in enumerate(self.elections):
            for j, election_2 in enumerate(self.elections):

                if i == j:
                    if self_distances:
                        ids.append((election_1, election_2))
                elif i < j:
                    ids.append((election_1, election_2))

        num_distances = len(ids)

        for t in range(num_threads):
            print('thread: ', t)
            sleep(0.1)
            start = int(t * num_distances / num_threads)
            stop = int((t + 1) * num_distances / num_threads)
            thread_ids = ids[start:stop]

            threads[t] = Thread(target=metr.single_thread, args=(self, distances, times, thread_ids, t))
            threads[t].start()

        for t in range(num_threads):
            threads[t].join()

        if self.store:
            path = os.path.join(os.getcwd(), "experiments", self.experiment_id, "distances",
                                str(distance_name) + ".csv")

            with open(path, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                writer.writerow(["election_id_1", "election_id_2", "distance"])

                for i, election_1 in enumerate(self.elections):
                    for j, election_2 in enumerate(self.elections):
                        if i < j:
                            distance = str(distances[election_1][election_2])
                            writer.writerow([election_1, election_2, distance])

        self.distances = distances
        self.times = times

    def create_structure(self):

        # PREPARE STRUCTURE

        if not os.path.isdir("experiments/"):
            os.mkdir(os.path.join(os.getcwd(), "experiments"))

        if not os.path.isdir("images/"):
            os.mkdir(os.path.join(os.getcwd(), "images"))

        if not os.path.isdir("trash/"):
            os.mkdir(os.path.join(os.getcwd(), "trash"))

        try:
            os.mkdir(os.path.join(os.getcwd(), "experiments", self.experiment_id))

            os.mkdir(os.path.join(os.getcwd(), "experiments", self.experiment_id, "distances"))
            os.mkdir(os.path.join(os.getcwd(), "experiments", self.experiment_id, "features"))
            os.mkdir(os.path.join(os.getcwd(), "experiments", self.experiment_id, "coordinates"))
            os.mkdir(os.path.join(os.getcwd(), "experiments", self.experiment_id, "elections"))
            os.mkdir(os.path.join(os.getcwd(), "experiments", self.experiment_id, "matrices"))

            # PREPARE MAP.CSV FILE

            path = os.path.join(os.getcwd(), "experiments", self.experiment_id, "map.csv")
            with open(path, 'w') as file_csv:
                file_csv.write(
                    "size,num_candidates,num_voters,election_model,param_1,param_2,color,alpha,label,marker,show\n")
                file_csv.write("3,10,100,impartial_culture,0,0,black,1,Impartial Culture,o,t\n")
                file_csv.write("3,10,100,iac,0,0,black,0.7,IAC,o,t\n")
                file_csv.write("3,10,100,conitzer,0,0,limegreen,1,SP by Conitzer,o,t\n")
                file_csv.write("3,10,100,walsh,0,0,olivedrab,1,SP by Walsh,o,t\n")
                file_csv.write("3,10,100,spoc_conitzer,0,0,DarkRed,0.7,SPOC,o,t\n")
                file_csv.write("3,10,100,group-separable,0,0,blue,1,Group-Separable,o,t\n")
                file_csv.write("3,10,100,single-crossing,0,0,purple,0.6,Single-Crossing,o,t\n")
                file_csv.write("3,10,100,1d_interval,0,0,DarkGreen,1,1D Interval,o,t\n")
                file_csv.write("3,10,100,2d_disc,0,0,Green,1,2D Disc,o,t\n")
                file_csv.write("3,10,100,3d_cube,0,0,ForestGreen,0.7,3D Cube ,o,t\n")
                file_csv.write("3,10,100,2d_sphere,0,0,black,0.2,2D Sphere,o,t\n")
                file_csv.write("3,10,100,3d_sphere,0,0,black,0.4,3D Sphere,o,t\n")
                file_csv.write("3,10,100,urn_model,0.1,0,yellow,1,Urn model 0.1,o,t\n")
                file_csv.write("3,10,100,norm-mallows,0.5,0,blue,1,Norm-Mallows 0.5,o,t\n")
                file_csv.write("3,10,100,urn_model,0,0,orange,1,Urn model (gamma),o,t\n")
                file_csv.write("3,10,100,norm-mallows,0,0,cyan,1,Norm-Mallows (uniform),o,t\n")
                file_csv.write("1,10,100,identity,0,0,blue,1,Identity,x,t\n")
                file_csv.write("1,10,100,uniformity,0,0,black,1,Uniformity,x,t\n")
                file_csv.write("1,10,100,antagonism,0,0,red,1,Antagonism,x,t\n")
                file_csv.write("1,10,100,stratification,0,0,green,1,Stratification,x,t\n")
                file_csv.write("1,10,100,walsh_matrix,0,0,olivedrab,1,Walsh Matrix,x,t\n")
                file_csv.write("1,10,100,conitzer_matrix,0,0,limegreen,1,Conitzer Matrix,x,t\n")
                file_csv.write("1,10,100,single-crossing_matrix,0,0,purple,0.6,Single-Crossing Matrix,x,t\n")
                file_csv.write("1,10,100,gs_caterpillar_matrix,0,0,green,1,GS Caterpillar Matrix,x,t\n")
                file_csv.write("3,10,100,unid,4,0,blue,1,UNID,3,f\n")
                file_csv.write("3,10,100,anid,4,0,black,1,ANID,3,f\n")
                file_csv.write("3,10,100,stid,4,0,black,1,STID,3,f\n")
                file_csv.write("3,10,100,anun,4,0,black,1,ANUN,3,f\n")
                file_csv.write("3,10,100,stun,4,0,black,1,STUN,3,f\n")
                file_csv.write("3,10,100,stan,4,0,red,1,STAN,3,f\n")
        except:
            # print("Experiment already exists!")
            pass

    def add_elections_to_experiment(self, with_matrices=False):
        """ Import elections from a file """

        elections = {}

        for family_id in self.families:
            if self.families[family_id].single_election:
                election_id = family_id
                election = Election(self.experiment_id, election_id, with_matrix=with_matrices)
                elections[election_id] = election
            else:
                for j in range(self.families[family_id].size):
                    election_id = family_id + '_' + str(j)
                    election = Election(self.experiment_id, election_id, with_matrix=with_matrices)
                    elections[election_id] = election

        return elections

    def add_distances_to_experiment(self):
        distances = self.import_my_distances()
        return distances

    def add_coordinates_to_experiment(self):
        coordinates = self.import_cooridnates()
        return coordinates

    def embed(self, attraction_factor=1, algorithm='spring', num_iterations=1000, distance_name='emd-positionwise'):
        num_elections = len(self.distances)

        X = np.zeros((num_elections, num_elections))

        for i, election_1_id in enumerate(self.distances):
            for j, election_2_id in enumerate(self.distances):
                if i < j:
                    if self.distances[election_1_id][election_2_id] == 0:
                        self.distances[election_1_id][election_2_id] = 0.01
                    if algorithm == 'spring':
                        X[i][j] = 1. / self.distances[election_1_id][election_2_id]
                    else:
                        X[i][j] = self.distances[election_1_id][election_2_id]
                    X[i][j] = X[i][j] ** attraction_factor
                    X[j][i] = X[i][j]

        dt = [('weight', float)]
        X = X.view(dt)
        G = nx.from_numpy_matrix(X)

        if algorithm == 'spring':
            my_pos = nx.spring_layout(G, iterations=num_iterations, dim=2)
        elif algorithm in {'mds', 'MDS'}:
            my_pos = MDS(n_components=2).fit_transform(X)
        elif algorithm in {'tsne', 'TSNE'}:
            my_pos = TSNE(n_components=2).fit_transform(X)

        points = {}
        for i, election_id in enumerate(self.distances):
            points[election_id] = [my_pos[i][0], my_pos[i][1]]

        # todo: store to file
        if self.store:
            file_name = os.path.join(os.getcwd(), "experiments", self.experiment_id,
                                     "coordinates", distance_name + "_2d_a" + str(float(attraction_factor)) + ".csv")

            with open(file_name, 'w', newline='') as csvfile:

                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow(["election_id", "x", "y"])

                ctr = 0
                for election_model_id in self.families:
                    for j in range(self.families[election_model_id].size):
                        a = election_model_id + '_' + str(j)
                        x = round(my_pos[ctr][0], 5)
                        y = round(my_pos[ctr][1], 5)
                        writer.writerow([a, x, y])
                        ctr += 1

        self.coordinates = points

    def print_map_tmp(self, group_by=None, saveas=None):
        if group_by is None:

            for election_id in self.coordinates:
                plt.scatter(self.coordinates[election_id][0], self.coordinates[election_id][1], label=election_id)
            plt.legend()
            plt.show()

        else:

            for color in group_by:
                X = []
                Y = []
                for election_id in group_by[color]:
                    X.append(self.coordinates[election_id][0])
                    Y.append(self.coordinates[election_id][1])
                plt.scatter(X, Y, label=group_by[color][0], color=color)
            plt.legend()
            if saveas is not None:
                file_name = saveas + ".png"
                path = os.path.join(os.getcwd(), file_name)
                plt.savefig(path, bbox_inches='tight')
            plt.show()

    def print_map(self, dim='2d', **kwargs):
        """ Print the two-dimensional embedding of multi-dimensional map of the elections """
        if dim == '2d':
            self.print_map_2d(**kwargs)
        elif dim == '3d':
            self.print_map_3d(**kwargs)

    def print_map_2d(self, mask=False, mixed=False, fuzzy_paths=True, xlabel=None,
                     angle=0, reverse=False, update=False, feature=None, attraction_factor=1, axis=False,
                     distance_name="emd-positionwise", guardians=False, ticks=None,
                     title=None,
                     saveas=None, show=True, ms=20, normalizing_func=None, xticklabels=None, cmap=None,
                     ignore=None, marker_func=None, tex=False, black=False, legend=True, levels=False, tmp=False):

        self.compute_points_by_families()

        if angle != 0:
            self.rotate(angle)

        if reverse:
            self.reverse()

        if update:
            self.update()

        if cmap is None:
            cmap = pr.custom_div_cmap()

        if feature is not None:
            fig = plt.figure(figsize=(6.4, 4.8 + 0.48))
        else:
            fig = plt.figure()
        ax = fig.add_subplot()

        if not axis:
            plt.axis('off')

        # COLORING
        if feature is not None:
            pr.color_map_by_feature(experiment=self, fig=fig, ax=ax, feature=feature,
                                    normalizing_func=normalizing_func, marker_func=marker_func,
                                    xticklabels=xticklabels, ms=ms, cmap=cmap, ticks=ticks)
        else:
            pr.basic_coloring(experiment=self, ax=ax, ms=ms)

        # BACKGROUND
        pr.basic_background(ax=ax, values=feature, legend=legend, saveas=saveas, xlabel=xlabel,
                            title=title)

        if tex:
            pr.saveas_tex(saveas=saveas)

        if show:
            plt.show()

    def print_map_3d(self, mask=False, mixed=False, fuzzy_paths=True, xlabel=None,
                     angle=0, reverse=False, update=False, feature=None, attraction_factor=1, axis=False,
                     distance_name="emd-positionwise", guardians=False,  ticks=None,
                     title=None,
                     saveas="map_2d", show=True, ms=20, normalizing_func=None, xticklabels=None, cmap=None,
                     ignore=None, marker_func=None, tex=False, black=False, legend=True, levels=False, tmp=False):

        self.compute_points_by_families()

        # if angle != 0:
        #     self.rotate(angle)

        # if reverse:
        #     self.reverse()

        # if update:
        #     self.update()

        if cmap is None:
            cmap = pr.custom_div_cmap()

        if feature is not None:
            fig = plt.figure(figsize=(6.4, 4.8 + 0.48))
        else:
            fig = plt.figure()
        ax = fig.add_subplot()

        if not axis:
            plt.axis('off')

        # COLORING
        if feature is not None:
            pr.color_map_by_feature(experiment=self, fig=fig, ax=ax, feature=feature,
                                    normalizing_func=normalizing_func, marker_func=marker_func,
                                    xticklabels=xticklabels, ms=ms, cmap=cmap, ticks=ticks)
        else:
            pr.basic_coloring(experiment=self, ax=ax, ms=ms)

        # BACKGROUND
        pr.basic_background(ax=ax, values=feature, legend=legend, saveas=saveas, xlabel=xlabel,
                            title=title)

        if tex:
            pr.saveas_tex(saveas=saveas)

        if show:
            plt.show()



    # def add_matrices_to_experiment(self):
    #     """ Import elections from a file """
    #
    #     matrices = {}
    #     vectors = {}
    #
    #     for family_id in self.families:
    #         for j in range(self.families[family_id].size):
    #             election_id = family_id + '_' + str(j)
    #             matrix = self.import_matrix(election_id)
    #             matrices[election_id] = matrix
    #             vectors[election_id] = matrix.transpose()
    #
    #     return matrices, vectors

    # def import_matrix(self, election_id):
    #
    #     file_name = election_id + '.csv'
    #     path = os.path.join(os.getcwd(), "experiments", self.experiment_id, 'matrices', file_name)
    #     num_candidates = self.elections[election_id].num_candidates
    #     matrix = np.zeros([num_candidates, num_candidates])
    #
    #     with open(path, 'r', newline='') as csv_file:
    #         reader = csv.DictReader(csv_file, delimiter=',')
    #         for i, row in enumerate(reader):
    #             for j, candidate_id in enumerate(row):
    #                 matrix[i][j] = row[candidate_id]
    #     return matrix

    def import_controllers(self, ignore=None):
        """ Import controllers from a file """

        families = {}

        path = os.path.join(os.getcwd(), "experiments", self.experiment_id, 'map.csv')
        file_ = open(path, 'r')

        header = [h.strip() for h in file_.readline().split(',')]
        reader = csv.DictReader(file_, fieldnames=header)

        starting_from = 0
        for row in reader:

            election_model = None
            color = None
            label = None
            param_1 = None
            param_2 = None
            alpha = None
            size = None
            marker = None
            num_candidates = None
            num_voters = None

            if 'election_model' in row.keys():
                election_model = str(row['election_model']).strip()

            if 'color' in row.keys():
                color = str(row['color']).strip()

            if 'label' in row.keys():
                label = str(row['label'])

            if 'param_1' in row.keys():
                param_1 = float(row['param_1'])

            if 'param_2' in row.keys():
                param_2 = float(row['param_2'])

            if 'alpha' in row.keys():
                alpha = float(row['alpha'])

            if 'size' in row.keys():
                size = int(row['size'])

            if 'marker' in row.keys():
                marker = str(row['marker']).strip()

            if 'num_candidates' in row.keys():
                num_candidates = int(row['num_candidates'])

            if 'num_voters' in row.keys():
                num_voters = int(row['num_voters'])

            show = True
            if row['show'].strip() != 't':
                show = False

            family_id = election_model + '_' + str(num_candidates) + '_' + str(num_voters)
            if election_model in {'urn_model', 'norm-mallows', 'mallows', 'norm-mallows_matrix'} and param_1 != 0:
                family_id += '_' + str(float(param_1))
            if election_model in {'norm-mallows', 'mallows'} and param_2 != 0:
                family_id += '__' + str(float(param_2))

            families[family_id] = Family(election_model=election_model, family_id=family_id,
                                         param_1=param_1, param_2=param_2, label=label,
                                         color=color, alpha=alpha, show=show, size=size, marker=marker,
                                         starting_from=starting_from,
                                         num_candidates=num_candidates, num_voters=num_voters)
            starting_from += size

        self.num_families = len(families)
        self.num_elections = sum([families[family_id].size for family_id in families])
        self.main_order = [i for i in range(self.num_elections)]

        if ignore is None:
            ignore = []

        # todo: update this part of code
        # ctr = 0
        # for family_id in families:
        #     resize = 0
        #     for j in range(families[family_id].size):
        #         if self.main_order[ctr] >= self.num_elections or self.main_order[ctr] in ignore:
        #             resize += 1
        #         ctr += 1
        #     families[family_id].size -= resize

        file_.close()
        return families

    # def import_order(self, main_order_name):
    #     """Import precomputed order of all the elections from a file."""
    #
    #     if main_order_name == 'default':
    #         main_order = [i for i in range(self.num_elections)]
    #
    #     else:
    #         file_name = os.path.join(os.getcwd(), "experiments", self.experiment_id, "results", "orders", main_order_name + ".txt")
    #         file_ = open(file_name, 'r')
    #         file_.readline()  # skip this line
    #         all_elections = int(file_.readline())
    #         file_.readline()  # skip this line
    #         main_order = []
    #
    #         for w in range(all_elections):
    #             main_order.append(int(file_.readline()))
    #
    #     return main_order

    def import_points(self, ignore=None):
        """ Import from a file precomputed coordinates of all the points -- each point refer to one election """

        if ignore is None:
            ignore = []

        points = {}
        path = os.path.join(os.getcwd(), "experiments", self.experiment_id,
                            "coordinates", self.distance_name + "_2d_a" + str(float(self.attraction_factor)) + ".csv")

        with open(path, 'r', newline='') as csv_file:
            reader = csv.DictReader(csv_file, delimiter=',')
            ctr = 0
            # print(path)
            for row in reader:
                if self.main_order[ctr] < self.num_elections and self.main_order[ctr] not in ignore:
                    points[row['election_id']] = [float(row['x']), float(row['y'])]
                ctr += 1

        return len(points), points

    def import_cooridnates(self, ignore=None):
        """ Import from a file precomputed coordinates of all the points -- each point refer to one election """

        if ignore is None:
            ignore = []

        points = {}
        path = os.path.join(os.getcwd(), "experiments", self.experiment_id,
                            "coordinates", self.distance_name + "_2d_a" + str(float(self.attraction_factor)) + ".csv")

        with open(path, 'r', newline='') as csv_file:
            reader = csv.DictReader(csv_file, delimiter=',')
            ctr = 0
            for row in reader:
                if self.main_order[ctr] < self.num_elections and self.main_order[ctr] not in ignore:
                    points[row['election_id']] = [float(row['x']), float(row['y'])]
                ctr += 1

        return points

    def compute_points_by_families(self):
        """ Group all points by their families """

        COLORS = ['blue', 'green', 'black', 'red', 'orange', 'purple', 'brown', 'lime', 'cyan', 'grey']

        if self.points_by_families is None:
            points_by_families = {}

        ### NEW ###
        if self.families is None:
            self.families = {}
            for i, election_id in enumerate(self.elections):
                ele = self.elections[election_id]
                # print(ele)
                election_model = ele.election_model
                family_id = election_model
                # param_1 = 0
                # param_2 = 0
                label = election_id
                color = COLORS[int(i % len(COLORS))]
                # todo: if there are more elections than len(COLORS) lower the alpha
                alpha = 1.
                # show = True
                # size = 1
                # marker = 'o'
                # starting_from = 0
                num_candidates = ele.num_candidates
                num_voters = ele.num_voters

                self.families[election_id] = Family(election_model=election_model, family_id=family_id,
                                                    label=label, alpha=alpha,
                                                    color=color,
                                                    num_candidates=num_candidates, num_voters=num_voters)

            for family_id in self.families:
                points_by_families[family_id] = [[] for _ in range(2)]
                points_by_families[family_id][0].append(self.coordinates[family_id][0])
                points_by_families[family_id][1].append(self.coordinates[family_id][1])

        ### ### ###
        else:
            for family_id in self.families:
                # print(family_id)
                points_by_families[family_id] = [[] for _ in range(2)]

                for i in range(self.families[family_id].size):
                    # print(i)
                    # print(self.coordinates)
                    if self.families[family_id].single_election:
                        election_id = family_id
                    else:
                        election_id = family_id + '_' + str(i)
                    points_by_families[family_id][0].append(self.coordinates[election_id][0])
                    points_by_families[family_id][1].append(self.coordinates[election_id][1])

        self.points_by_families = points_by_families

    def compute_feature(self, name=None, attraction_factor=1):

        # values = []
        feature_dict = {}

        for election_id in self.elections:
            feature = features.get_feature(name)
            election = self.elections[election_id]
            # print(election_id, election)
            if name in {'avg_distortion_from_guardians', 'worst_distortion_from_guardians'}:
                value = feature(self, election_id)
            else:
                value = feature(election)
            # values.append(value)
            feature_dict[election_id] = value

        if self.store:
            path = os.path.join(os.getcwd(), "experiments", self.experiment_id, "features", str(name) + '.csv')
            with open(path, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                writer.writerow(["election_id", "value"])
                for key in feature_dict:
                    writer.writerow([key, feature_dict[key]])

        self.features[name] = feature_dict
        return feature_dict

    def get_distance(self, i, j):
        """ Compute Euclidean distance in two-dimensional space"""

        distance = 0.
        for d in range(2):
            distance += (self.coordinates[i][d] - self.coordinates[j][d]) ** 2

        return math.sqrt(distance)

    def rotate(self, angle):
        """ Rotate all the points by a given angle """

        for i in range(self.num_elections):
            self.coordinates[i][0], self.coordinates[i][1] = self.rotate_point(0.5, 0.5, angle, self.coordinates[i][0],
                                                                               self.coordinates[i][1])

        self.points_by_families = self.compute_points_by_families()

    def reverse(self):
        """ Reverse all the points"""

        for i in range(self.num_elections):
            self.coordinates[i][0] = self.coordinates[i][0]
            self.coordinates[i][1] = -self.coordinates[i][1]

        self.points_by_families = self.compute_points_by_families()

    def update(self):
        """ Save current coordinates of all the points to the original file"""

        path = os.path.join(os.getcwd(), "experiments", self.experiment_id,
                            "points", self.distance_name + "_2d_a" + str(self.attraction_factor) + ".csv")

        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(["id", "x", "y"])

            for i in range(self.num_elections):
                x = round(self.coordinates[i][0], 5)
                y = round(self.coordinates[i][1], 5)
                writer.writerow([i, x, y])

    @staticmethod
    def rotate_point(cx, cy, angle, px, py):
        """ Rotate two-dimensional point by an angle """

        s = math.sin(angle)
        c = math.cos(angle)
        px -= cx
        py -= cy
        x_new = px * c - py * s
        y_new = px * s + py * c
        px = x_new + cx
        py = y_new + cy

        return px, py

    def import_distances(self, self_distances=False, distance_name='emd-positionwise'):
        """ Import precomputed distances between each pair of elections from a file """

        file_name = str(distance_name) + '.csv'
        path = os.path.join(os.getcwd(), "experiments", self.experiment_id, "distances", file_name)
        num_points = self.num_elections
        num_distances = int(num_points * (num_points - 1) / 2)

        hist_data = {}
        std = [[0. for _ in range(num_points)] for _ in range(num_points)]

        for family_id in self.families:
            for j in range(self.families[family_id].size):
                election_id = family_id + '_' + str(j)
                hist_data[election_id] = {}

        with open(path, 'r', newline='') as csv_file:
            reader = csv.DictReader(csv_file, delimiter=',')

            for row in reader:
                election_id_1 = row['election_id_1']
                election_id_2 = row['election_id_2']
                hist_data[election_id_1][election_id_2] = float(row['distance'])
                hist_data[election_id_2][election_id_1] = hist_data[election_id_1][election_id_2]

        # todo: add self-distances
        # for a in range(num_points):
        #     limit = a+1
        #     if self_distances:
        #         limit = a
        #     for b in range(limit, num_points):
        #         line = file_.readline()
        #         line = line.split(' ')
        #         hist_data[a][b] = float(line[2])
        #
        #         # tmp correction for discrete distance
        #         if distance_name == 'discrete':
        #             hist_data[a][b] = self.families[0].size - hist_data[a][b]   # todo: correct this
        #
        #
        #         hist_data[b][a] = hist_data[a][b]
        #
        #         if distance_name == 'voter_subelection':
        #             std[a][b] = float(line[3])
        #             std[b][a] = std[a][b]

        return num_distances, hist_data, std

    def import_my_distances(self, self_distances=False, distance_name='emd-positionwise'):
        """ Import precomputed distances between each pair of elections from a file """

        file_name = str(distance_name) + '.csv'
        path = os.path.join(os.getcwd(), "experiments", self.experiment_id, "distances", file_name)
        distances = {}

        for family_id in self.families:
            for j in range(self.families[family_id].size):
                election_id = family_id + '_' + str(j)
                distances[election_id] = {}

        with open(path, 'r', newline='') as csv_file:
            reader = csv.DictReader(csv_file, delimiter=',')

            for row in reader:
                election_id_1 = row['election_id_1']
                election_id_2 = row['election_id_2']
                distances[election_id_1][election_id_2] = float(row['distance'])
                distances[election_id_2][election_id_1] = distances[election_id_1][election_id_2]
        return distances