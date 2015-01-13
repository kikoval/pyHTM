import numpy as np
from operator import itemgetter

import pyhtm.utils as utils
from pyhtm.support.getsetstate import (GetSomeVars, UpdateMembers)


class SpatialPooler:

    def __init__(self, algorithm='gaussian', max_distance=0.01, sigma=0.5,
                 max_coincidence_count=512, rare_coincidence_threshold=0,
                 ignore_background_pattern=True, background_color=1,
                 debug=False):
        """SpatialPooler during learning performs the clustering of input data
        into predefined number of clusters and during inference finds the
        closest cluster for the new data.

        Args:
            algorithm: name of the inference algorithm (gaussian, dot, product, sum)
            max_distance: learning parameter for clustering
            max_coincidence_count: maximal number of stored coincidences
            rare_coincidence_threshold: minimal number of occurences during
                                        training for a coincidence to keep
            debug: whether to show debug messages
        """
        self.algorithm = algorithm
        self.max_distance = max_distance
        # used in gaussian spatial inference
        self.sigma_sq = 2 * sigma**2 if sigma is not None else None
        self.max_coincidence_count = max_coincidence_count
        # minimum num. of occurences to keep
        self.rare_coincidence_threshold = rare_coincidence_threshold
        # self.sparsify = sparsify
        self.debug = debug

        self.coincidences = dict(np.array([]))  # id => coincidence
        self.coincidences_stats = dict()  # id => num. of occurences

        self.ignore_background_pattern = ignore_background_pattern
        self.background_color = background_color

        self.vectorized_gaussian = np.vectorize(self._gaussian)  # for faster inference

    def _euclidean_distance(self, x, y):
        """Compute Euclidean distance between two vectors.

        Args:
            x: a numpy vector
            y: a numpy vector

        Returns:
            a float number
        """
        return np.sum((x - y) ** 2) ** .5

    def _widx_distance(self, x, y):
        """Compute Hamming distance between two (binary) vectors.

        Args:
            x: a numpy vector
            y: a numpy vector

        Returns:
            an integer representing the number of positions at which are
            the vectors different
        """
        return np.sum((x - y) != 0)

    def _distance(self, x, y):
        """Compute Euclidean or Hamming distance (depending on inference
        algorithm) between two vectors.

        Args:
            x: a numpy vector
            y: a numpy vector

        Returns:
            distance between x and y
        """
        if self.algorithm == 'gaussian':
            return self._euclidean_distance(x, y)
        else:
            return self._widx_distance(x, y)

    def _compute_dist_to_coincidences(self, pattern=None):
        # TODO optimize
        return [(i, self._distance(pattern, c)) for i, c in
                self.coincidences.items()]

    def _get_coincedence_id(self, pattern):
        """Find coincidence with the minimal distance.

        Args:
            pattern: input message from the input or layer below

        Returns:
            id of the closest coincidence
        """
        dist = self._compute_dist_to_coincidences(pattern)
        return min(dist, key=itemgetter(1))[0]  # sort by dist, return ID

    def learn(self, evidence):
        """Quantize data into 'max_coincidence_count' clusters with maximal size
        of 'max_distance'.

        Args:
            evidence: a numpy vector to be learned
        """
        # input checking
        if not utils.is_list(evidence) and not evidence:
            raise ValueError("'evidence' has incorrect value")

        # don't learn empty patterns
        if self.ignore_background_pattern:
            is_empty = np.min(evidence) == np.max(evidence) == self.background_color
            if is_empty:
                return False

        if self.algorithm in ('product', 'sum'):
            # keep only the indices of the maximal activation for each child
            evidence = utils.get_max_indices(evidence)

        # if there is no memorized coincidence, store it
        if len(self.coincidences) == 0:
            self.coincidences[0] = evidence
            self.coincidences_stats[0] = 1
        else:
            # store the evidence iff d(evicence, coincidences) > maxDistance
            # dist is a list of tuples [ (index, distance), ...]
            dist = self._compute_dist_to_coincidences(evidence)

#            if self.debug and min(dist, key=itemgetter(1))[1] > 0:
#                print "-> The distance to the closest coincidence is %.2f" % min(dist, key=itemgetter(1))[1]

            codebook_is_not_full = len(self.coincidences) < self.max_coincidence_count
            nearest_coincidence = min(dist, key=itemgetter(1))  # tuple (id, distance)

            if codebook_is_not_full and nearest_coincidence[1] > self.max_distance:
                self.coincidences[len(self.coincidences)] = evidence  # new coincidence
                self.coincidences_stats[len(self.coincidences) - 1] = 1
#                if self.debug: print "-> New coincidence added, count = %d" % len(self.coincidences)
            else:
                # Increase stats for the closest coincidence
                # nearest_coincidence[0] is the index of the coincidence
                self.coincidences_stats[nearest_coincidence[0]] += 1

    def _add_coincidence(self, evidence):
        self.coincidences[len(self.coincidences)] = evidence  # new coincidence
        self.coincidences_stats[len(self.coincidences) - 1] = 1

    def _remove_coincidence(self, coincidence_id):
        self.coincidences.pop(coincidence_id)
        self.coincidences_stats.pop(coincidence_id)

    def finalize_learning(self):
        """
        1. Remove rare coincidences (if enabled)
        2. Add background pattern (if applicable)
        3. Optimize coincidence data structure for faster inference
        """
        # TODO set the threshold to be relative to mean/max
        # 1.
        if self.rare_coincidence_threshold > 0:
            for c_id in list(self.coincidences.keys()):
                if self.coincidences_stats[c_id] < self.rare_coincidence_threshold:
                    self._remove_coincidence(c_id)

        # 2.
        if self.ignore_background_pattern:
            empty_pattern = np.ones(self.coincidences[0].shape) * self.background_color
            self._add_coincidence(empty_pattern)
            self.max_coincidence_count += 1

        # 3.
        # convert dict() into a list so it can be easier to work with them when
        # some of the coincidences were removed
#        self.coincidences = self.coincidences.values()
        self.coincidences_stats = list(self.coincidences_stats.values())

        if self.debug: print("Number of coincidences %d" % len(self.coincidences))
        if self.debug and self.max_coincidence_count < len(self.coincidences):
            print("Warning: only %d coincidences were stored. %d was requested."
                  % (len(self.coincidences), self.max_coincidence_count))

        # creating data matrix for faster distance computation in inference
        self.coincidences_matrix = np.vstack(list(self.coincidences.values()))

    def _gaussian(self, x):
        """Apply Gaussian-like weighting.
        Args:
            x: a float number

        Returns:
            function value of f(x) = e^\frac{-x^2}{2*sigma^2}
        """
        return np.exp(- x**2 / self.sigma_sq)

    def _product_inference(self, coincidence, evidence):
        """Compute probability of evidence given coincidence. The evidence
        consists of concatenated child outputs which are considered to be
        independent.

        Args:
            coincidence: a vector of indices of max values in message from each child
            evidence: a probability vector

        Returns:
            a float number proportional to P(evidence|coincidence)

        Example:
            coincedence = [2, 3, 1]
            evidence = [0, 0, 0.8, 0,  0, 0, 0, 0.9,  0, 1, 0, 0]
            >> child_length = 4
            >> y = 0.8 * 0.9 * 1 = 0.72
        """
        child_length = int(len(evidence) / len(coincidence))
        y = 1.0
        for i, w_id in enumerate(coincidence):
            y *= evidence[i*child_length + w_id]
        return y

    def _sum_inference(self, coincidence, evidence):
        """Compute probability of evidence given coincidence. The evidence
        consists of concatenated child outputs which are considered to be
        independent.

        Args:
            coincidences: a vector of indices
            evidence: a probability vector

        Returns:
            a float number proportional to P(evidence|coincidence)
        """
        child_length = int(len(evidence) / len(coincidence))
        y = 0.0
        for i, w_id in enumerate(coincidence):
            y += evidence[i*child_length + w_id]
        return y

    def infer(self, evidence, is_reset=False):
        """Compute output vector p(evidence|C). Gaussian, dot and product
        inference algorithms are set in the constructor.

        Args:
            evidence: a vector

        Returns:
            A vector representing the activations of coincidences given
            the evidence. The vector is normalized to sum to 1.
        """
        if is_reset:
            return
        # TODO ratat statistiku coincidencii este pocas temporalneho ucenia?
        y = np.ones(len(self.coincidences))

        if self.algorithm == 'gaussian':
            # computing distances between evidence and all coincidences
            # TODO refactor using self._distance and np.vectorize
            dist = (self.coincidences_matrix - evidence) ** 2
            dist = np.sum(dist, axis=1) ** .5
            y = self.vectorized_gaussian(dist)

            # sparsifying the output
#            num_max_values = int(len(y) * .05)  # 5%
#            z = y.argsort()[-3:]  # 3 max values
##            closest_id = np.argmax(y)
#            mask = np.zeros(len(y))
#            for i in z:
#                mask[i] = 1;
#            y = y * mask

        if self.algorithm == 'dot':
            for i, c in enumerate(self.coincidences.values()):
                y[i] = np.dot(evidence, c)

        # Equation A.2 from George's thesis, it assumes that the evidences from
        # the children can be combined independently given the node's coincidences
        #
        # Example:
        #   a node has two children with three temporal groups, the concatinated
        #   input from those children is i=[0.8, 0.1, 0.1, 0.5, 0.2, 0.3],
        #   eg. input from the 1st children is [0.8, 0.1, 0.1] and from the 2nd
        #   [0.5, 0.2, 0.3], a stored coincidence is [0, 2], the computation
        #   would be i[0] * i[2] = 0.8 * 0.2 = 0.16

        if self.algorithm == 'product':
            for i, c in enumerate(self.coincidences.values()):
                y[i] = self._product_inference(c, evidence)

        if self.algorithm == 'sum':
            for i, c in enumerate(self.coincidences.values()):
                y[i] = self._sum_inference(c, evidence)

        return utils.normalize_to_one(y)

    def __getstate__(self):
        return GetSomeVars(self, ('algorithm', 'coincidences', 'max_distance',
                            'sigma_sq', 'max_coincidence_count', 'debug',
                            'coincidences_stats', 'rare_coincidence_threshold',
                            'max_distance', 'coincidences_matrix'))

    def __setstate__(self, state):
        UpdateMembers(self, state)
        self.vectorized_gaussian = np.vectorize(self._gaussian)
