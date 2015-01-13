import time
import numpy as np

from pyhtm.nodes.imagesensor import ImageSensor
from pyhtm.level import HTMLevel
import pyhtm.utils as utils
import pyhtm.support.visualize as visualize


class HTM:
    """The main class covering the creation, learning and inference of the
    Hierachical temporal memory network.

    Attributes:
        sensor: retrieves the input, eg. ImageSensor
        levels: list of HTMLevel objects
        debug: boolean indicating whether to print debug messages
        show_graphs: boolean if to show visualization of codebook and
                     temporal groups
        verbose: boolean indicationg whether to print informative messages
        patch_size: patch dimmensions in pixels (width, height)
    """

    def __init__(self, sensor=None, debug=False, show_graphs=False, verbose=False):
        self.sensor = sensor
        self.debug = debug
        self.show_graphs = show_graphs
        self.verbose = verbose

        self.levels = []
        self.patch_size = []

        self.is_image_sensor = self.sensor is not None and isinstance(self.sensor, ImageSensor)

    def create_network(self, params):
        """Create network based on the provided parameters.

        Args:
            params: list of dicts sorted starting from the bottom level
        """
        if params is None:
            raise ValueError("Parameter 'params' should not be empty.")

        if self.verbose: print("Creating network...")

        for level_no in range(len(params)):

            if level_no == 0:
                # assuming rectangular arrangement of nodes
                if self.is_image_sensor:
                    self.patch_size = [int(self.sensor.width / params[0]['size'][0]),
                                       int(self.sensor.height / params[0]['size'][1])]
                    if self.verbose: print("Patch size is set to %s." % self.patch_size)
                prev_level_size = params[0]['size']
            else:
                prev_level_size = params[level_no - 1]['size']

            level = HTMLevel(level_no, params=params[level_no], debug=self.debug,
                             previous_level_size=prev_level_size)
            self.add_level(level)

        for level_no in range(len(self.levels)-1):
            self.levels[level_no].above_links = self.levels[level_no+1].links

        if self.verbose: print("=" * 40)

    def add_level(self, level):
        self.levels.append(level)

    def stats(self):
        print("Levels:", len(self.levels))
        for l in self.levels:
            print(l.stats())

    def _get_train_patterns(self, level, all_patterns=True):
        """Get an image from the sensor and either extracts one pattern or all
        of them.

        Args:
            level: level number
            all_patterns: A boolean indicating whether all patterns (image
                          patches) should be used

        Returns:
            A list of pattern(s).
        """
        data = self.sensor.compute()
        sensor_im, is_reset = data['data'], data['is_reset']
        if not isinstance(self.sensor, ImageSensor):
            return sensor_im, is_reset

        # if level.level_no == 0:
        #     visualize.show_image(sensor_im)

        # if level.level_no == 0 and level.node_cloning:
        if not all_patterns:
            # TODO how to pick a good spot?
            # central patch
            f = int(self.sensor.width/2 - self.patch_size[0]/2)
            t = int(self.sensor.width/2 + (self.patch_size[0] - self.patch_size[0]/2))
            pattern = sensor_im[f:t, f:t].reshape(np.prod(self.patch_size), 1)

            patterns = [pattern]  # * np.prod(self.levels[0].size)
        else:
            patterns = utils.extract_patches(sensor_im, self.patch_size, self.levels[0].overlap)

        return patterns, is_reset

    def learn(self, learning_params):
        """Do the learning of the HTM level by level according to the provided
        learning parameters.

        Args:
            learning_params: A dictionary of parameters for learning
        """
        for ln, level in enumerate(self.levels):
            if self.verbose: print("Learning level %d." % ln)

            # === Spatial pooling ===
            if self.verbose: print("Spatial learning...")
            start_time = time.clock()

            self.sensor.setParameter('explorer',
                                     learning_params[ln]['sp']['explorer'])
            self.sensor.seek(0)

            num_iterations = learning_params[ln]['sp']['numIterations']
            if self.verbose: print("  Num of iterations: %d" % num_iterations)

            for i in range(num_iterations):
                patterns, is_reset = self._get_train_patterns(level)
                if patterns is None:
                    break

                # infer through the levels below
                patterns = self.infer(patterns)

                level.do_spatial_learning(patterns)

            if self.show_graphs and level._is_first_level():
                visualize.show_codebook(level.nodes[0], self.patch_size)

            if self.verbose: print('  The spatial learning code ran for %.2fm'
                                   % ((time.clock() - start_time) / 60.))

            # === Temporal pooling ===
            if self.verbose: print("Temporal learning...")
            start_time = time.clock()

            self.sensor.setParameter('explorer',
                                     learning_params[ln]['tp']['explorer'])

            num_iterations = learning_params[ln]['tp']['numIterations']
            if self.verbose: print("  Num of iterations: %d" % num_iterations)

            # n_empty = 0
            get_all_patterns = not (level._is_first_level() and level.node_cloning)
            for i in range(num_iterations):
                patterns, is_reset = self._get_train_patterns(level, get_all_patterns)

                if is_reset:
                    # inference is not needed
                    level.do_temporal_learning(patterns, is_reset=True)
                else:
                    # infer through the levels below
                    patterns = self.infer(patterns)

                    level.do_temporal_learning(patterns)

            if self.verbose: print("Finalize learning...")
            level.finalize_learning()

            if self.show_graphs and level._is_first_level():
            #     visualize.show_TAM(level.nodes[0])
                visualize.show_temporal_groups(level.nodes[0], self.patch_size)

            if self.verbose: print('  The temporal learning code ran for %.2fm'
                                   % ((time.clock() - start_time) / 60.))

            if self.verbose: print("=" * 40)

    def infer(self, patterns, merge_output=True):
        """Perform inference on the provided patterns.

        Args:
            patterns: list of patterns
            merge_output: A boolean indicating whether the output should be
                          merged into one vector (list)

        Returns:
            an inference vector or a list of inference vectors
        """
        for i, level in enumerate(self.levels):
            if level.mode == 'infer':
                is_last = i == len(self.levels) - 1
                # if is_last: print "IS LAST"
                # if i==1: print "Level %d: infer output patterns lenght (%d)" % (i, len(patterns))
                    # (i < len(self.levels) - 1 and self.levels[i+1].mode is not 'infer')
                patterns = level.infer(patterns, merge_output=is_last and merge_output)
            else:
                # return the output of the last level in infer mode
                # the output is a list that was created by concatenating the
                # output of the child nodes
                return patterns

        return patterns

    def save(self, filename='network.pkl'):
        """Save (learned) HTM network to a binary pickle file.

        Args:
            filename: string
        """
        import pickle

        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.__dict__, file=f,
                            protocol=pickle.HIGHEST_PROTOCOL)
        except IOError:
            print("File is not writable.")

    def load(self, filename='network.pkl'):
        """Load HTM network data from a binary pickle file.

        Args:
            filename: string
        """
        import pickle

        try:
            with open(filename, 'rb') as f:
                res = pickle.load(f)
            self.__dict__.update(res)
        except IOError:
            print("File not found or readable.")
