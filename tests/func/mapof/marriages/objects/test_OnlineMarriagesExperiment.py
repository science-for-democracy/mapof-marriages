
import mapof.marriages as mapof


class TestOnlineMarriageExperiment:

    def setup_method(self):
        """Set up the experiment instance for each test."""
        self.experiment = mapof.prepare_online_marriages_experiment()

    def add_instance(self):
        self.experiment.add_instance(
            culture_id='impartial',
            num_agents=10
        )
        self.experiment.add_instance(
            culture_id='urn',
            num_agents=10,
            alpha=0.1
        )

    def add_families(self):
        """Helper method to add default families to the experiment."""
        self.experiment.add_family(
            culture_id='impartial',
            num_agents=10,
            size=10,
            color='green',
            marker='x',
            label='IC'
        )

        self.experiment.add_family(
            culture_id='urn',
            num_agents=10,
            size=10,
            alpha=0.1,
            color='blue',
            marker='o',
            label='Urn'
        )

    def test_experiment_creation(self):
        assert self.experiment is not None, "Experiment should be created successfully"

    def test_adding_instances(self):
        self.add_instance()
        assert self.experiment.num_instances == 2, "Two elections should be added"

    def test_adding_families(self):
        self.add_families()
        assert len(self.experiment.families) == 2, "Two families should be added"

    def test_computing_distances(self):
        self.add_instance()
        self.experiment.compute_distances(distance_id='l1-mutual_attraction')
        assert self.experiment.distances is not None, "Distances should be computed"

    def test_embedding(self):
        self.add_families()
        self.experiment.compute_distances(distance_id='l1-mutual_attraction')
        self.experiment.embed_2d(embedding_id='fr')
        assert self.experiment.coordinates is not None, "Embedding should be performed"

    def test_print_map(self):
        self.add_families()
        self.experiment.compute_distances(distance_id='l1-mutual_attraction')
        self.experiment.embed_2d(embedding_id='fr')
        self.experiment.print_map_2d(show=False)

    def test_compute_feature(self):
        self.add_families()
        self.experiment.compute_distances(distance_id='l1-mutual_attraction')
        self.experiment.embed_2d(embedding_id='fr')

        feature_id = 'summed_rank_minimal_matching'
        self.experiment.compute_feature(feature_id=feature_id)

    def test_print_map_colored_by_feature(self):
        self.add_families()
        self.experiment.compute_distances(distance_id='l1-mutual_attraction')
        self.experiment.embed_2d(embedding_id='fr')

        feature_id = 'summed_rank_minimal_matching'
        self.experiment.compute_feature(feature_id=feature_id)
        self.experiment.print_map_2d_colored_by_feature(
            show=False,
            feature_id=feature_id
        )





