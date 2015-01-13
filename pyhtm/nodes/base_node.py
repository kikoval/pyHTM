import numpy as np

from pyhtm.nodes.poolers import SpatialPooler, TemporalPooler
import pyhtm.utils as utils

default_params = {
    'spatialPoolerAlgorithm': 'gaussian',
    'maxDistance': 0.1,
    'sigma': 1.5,
    'maxCoincidenceCount': 128,
    'rareCoincidenceThreshold': 0,

    'temporalPoolerAlgorithm': 'maxProp',
    'transitionMemory': 4,
    'requestedGroupsCount': 20,
    'symmetrizeTAM': True,

    'ignoreBackgroundPattern': False,
    'backgroundColor': 1
}


class BaseNode:

    def __init__(self, params=None, debug=False):
        """
        Args:
            params: dict with parameters for spatial and temporal poolers
            debug: whether to print debug messages
        """
        params = utils.fill_default_params(params, default_params)
        self.debug = debug

        assert(params['maxCoincidenceCount'] > params['requestedGroupsCount'])
        if params['spatialPoolerAlgorithm'] == 'gaussian':
            assert(params['sigma'] >= 0)
        assert(params['maxDistance'] >= 0)

        self.sp = SpatialPooler(
            algorithm=params['spatialPoolerAlgorithm'],
            max_distance=params['maxDistance'],
            sigma=params['sigma'],
            max_coincidence_count=params['maxCoincidenceCount'],
            rare_coincidence_threshold=params['rareCoincidenceThreshold'],
            ignore_background_pattern=params['ignoreBackgroundPattern'],
            background_color=params['backgroundColor'],
            debug=self.debug
        )

        self.tp = TemporalPooler(
            algorithm=params['temporalPoolerAlgorithm'],
            transition_memory=params['transitionMemory'],
            requested_group_count=params['requestedGroupsCount'],
            symmetrizeTAM=params['symmetrizeTAM'],
            debug=self.debug
        )

        self.tp_first_run = True
        self.y = None  # spatial inference in fw direction
        self.fw_mesage = None  # forward output of the node

    def enable_debug(self):
        self.debug = True

    def do_spatial_learning(self, pattern):
        self.sp.learn(pattern)

    def do_temporal_learning(self, pattern, is_reset=False):
        if self.tp_first_run:
            self.sp.finalize_learning()
            self.tp.coincidences_stats = self.sp.coincidences_stats
            self.tp.coincidences_count = len(self.sp.coincidences_stats)
            self.tp_first_run = False

        self.tp.learn(self.sp.infer(pattern, is_reset), is_reset)

    def finalize_learning(self):
        self.tp.finalize_learning()

        if self.debug:
            print("  Codebook size after learning: %d"
                  % self.tp.coincidences_count)
            print("  Temporal group size after learning: %d"
                  % len(self.tp.temporal_groups))

    def infer(self, pattern):
        self.y = self.sp.infer(pattern)
        self.fw_mesage = self.tp.infer(self.y)  # feed-forward message
        return self.fw_mesage

    def print_status(self):
        print("Status of a HTM node")
        print("Receptive field: [%d,%d]" % (self.receptive_field[0],
                                            self.receptive_field[1]))
        print("# of coincidences", len(self.coincidences))
        print(self.coincidences_stats)
