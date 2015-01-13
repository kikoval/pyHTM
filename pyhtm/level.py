from operator import itemgetter
import numpy as np

from pyhtm.nodes import BaseNode
import pyhtm.utils as utils

default_params = {
    'size': [4, 4],
    'overlap': [0, 0],  # not implemented yet
    'nodeCloning': False,
    'sparsify': False  # not implemented yet
}


class HTMLevel:
    """Encapsulate one HTM level.
    """

    def __init__(self, level_no, params=None, previous_level_size=None,
                 debug=False):

        self.level_no = level_no
        self.debug = debug
        if previous_level_size is None:
            raise ValueError("previous_level_size must not be None.")

        self.mode = 'learn'  # learn or infer modes
        self.nodes = []  # nodes in the current level

        if params is not None:
            params = utils.fill_default_params(params, default_params)
            self.node_cloning = params['nodeCloning']
            self.size = params['size']  # num. of nodes [width, height]
            self.overlap = params['overlap']
            self.sparsify = params['sparsify']  # TODO

            # Create nodes
            if self.node_cloning:
                if self.debug:
                    print("Level %d: Creating one master node." % self.level_no)
                # create one master node
                self.nodes.append(BaseNode(params, debug=self.debug))
            else:
                if self.debug:
                    print("Level %d: Creating %d nodes." %
                          (self.level_no, np.prod(params['size'])))
                # create all nodes
                for i in range(np.prod(params['size'])):
                    self.nodes.append(BaseNode(params, debug=self.debug))
        else:
            raise ValueError("params must not be of type None.")

        # node_id: [list of node_ids from level-1]
        self.links = {}
        if self._is_first_level():
            # in 1-to-1 correnspondence to image patches
            for i in range(np.prod(self.size)):
                self.links[i] = [i]
        else:
            # TODO for rectangular regions
            # TODO factorize so it can be tested
            # every node in this level gets linked with equal number of nodes
            # in the previous level arraged in squares
            #
            # Example for [4, 4] -> [2,2]
            # 12 13 14 15      0: [0, 1, 4, 5]
            #  8  9 10 11      1: [2, 3, 6, 7]
            #  4  5  6  7  ->  2: [8, 9, 12, 13]
            #  0  1  2  3      3: [10, 11, 14, 15]
            #
            self.links = self._get_links(self.size, previous_level_size)

        self.above_links = None  # backwards mapping to self.link

    def _is_first_level(self):
        return self.level_no == 0

    def _get_links(self, size, previous_level_size):
        """Compute the links for all nodes in level to the nodes in the previous
        level.

        Args:
            size: list or tuple of number of nodes in a layer in a rectangle
                  matrix, e.g. (2, 2) is a square matrix with 2x2 nodes
            previous_level_size: list or tuple representing the size of the
                                 previous level
        Returns:
            a dictionary where for each index there is list of node IDs
        """
        # if size=[1, 1], there is only one node in a layer
        if size == [1, 1] or size == [1]:
            return {0: range(np.prod(previous_level_size))}

        # TODO add support for rectangular layers
        prev_lvl_size = previous_level_size[0]
        step = int(prev_lvl_size / size[0])
        i = -1
        links = {}
        for row in range(size[0]):
            for col in range(size[1]):
                i = i + 1
                links[i] = []
                for j in range(step):
                    links[i] += \
                    range(j*prev_lvl_size + col*step + row*step*prev_lvl_size,
                          j*prev_lvl_size + (col+1)*step + row*step*prev_lvl_size)
        return links

    def add_node(self, node):
        self.nodes.append(node)

    def do_spatial_learning(self, patterns, ignore_empty=False):
        """Learn spatial patterns.

        Args:
            patterns: a list of nodes' outputs from the previous layer
                      or the sensor
            ignore_empty: a boolean indicating whether to ignore empty patterns
        """
        # prepare patterns -- merge child outputs if necessary
        merged_patterns = []
        flatten = self.nodes[0].sp.algorithm is not 'product'
        for i in range(np.prod(self.size)):
            node_patterns = itemgetter(*self.links[i])(patterns)
            if flatten:
                node_patterns = np.asarray(utils.flatten_list(node_patterns))
            else:
                node_patterns = np.asarray(node_patterns)
            merged_patterns.append(node_patterns)

        if self.node_cloning:
            # one master node receives all the input
            for p in merged_patterns:
                # ignore empty patterns
                if self._is_first_level() and ignore_empty:
                    if np.min(p) == np.max(p):
                        continue
                self.nodes[0].do_spatial_learning(p)
        else:
            # each node gets its own input
            for n, p in zip(self.nodes, merged_patterns):
                # ignore empty patterns
                if self._is_first_level() and ignore_empty:
                    if np.min(p) == np.max(p):
                        continue
                n.do_spatial_learning(p)

    def do_temporal_learning(self, patterns, is_reset=False):
        for i, n in enumerate(self.nodes):
            if is_reset:
                n.do_temporal_learning(None, is_reset=True)
            else:
                node_patterns = itemgetter(*self.links[i])(patterns)
                node_patterns = np.asarray(utils.flatten_list(node_patterns))
                n.do_temporal_learning(node_patterns)

    def finalize_learning(self):
        """Finalize learning of all the nodes in the level and turn them into
        inference mode.
        When shared learning is turned on, copy the learned
        structures to all other nodes.
        """
        for n in self.nodes:
            n.finalize_learning()

        # switching to infer mode
        self.mode = 'infer'

        if self.node_cloning:
            import copy
            # copying node to fit the layer size
            for i in range(np.prod(self.size) - 1):
                self.nodes.append(copy.deepcopy(self.nodes[0]))

    def infer(self, patterns, merge_output=False):
        """Run infererence in each node in the level with the corresponding
        pattern from the previous level (input image).

        Args:
            patterns: a list of patterns from the level below or from the sensor
            merge_output: whether to return concatenated inference vector or a
                          list of inference vectors for each pattern

        Returns:
            an inference vector of a list of inference vectors
        """
        output = []  # nodes' outputs in the same order as in self.nodes
        for i, n in enumerate(self.nodes):
            node_patterns = itemgetter(*self.links[i])(patterns)
            node_patterns = np.asarray(utils.flatten_list(node_patterns))
            output.append(n.infer(node_patterns))

        if merge_output:
            # from [[output1], [output2]] makes [output1, output2]
            return np.asarray(utils.flatten_list(output))
        else:
            return np.asarray(output)

    def stats(self):
        print("Level", self.level)
        if len(self.nodes):
            for n in self.nodes:
                n.print_status()
