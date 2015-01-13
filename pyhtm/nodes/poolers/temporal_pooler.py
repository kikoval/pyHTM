import numpy as np
from operator import itemgetter

import pyhtm.utils as utils
#import pyhtm.support.visualize as visualize


class TemporalPooler:

    def __init__(self, coincidences_stats=None, algorithm='maxProp',
                 transition_memory=4, requested_group_count=64,
                 symmetrizeTAM=True, purge_garbage=False, debug=False):
        """
        Args:
        algorithm:
            When set to 'maxProp', computes a more peaked score for the group
            based on the current input only.
            When set to 'sumProp', computes a smoother score for the group
            based on the current input only.
            When set to 'tbi', computes a score using Time-Based Inference (TBI)
            which uses the current as well as past inputs (not implemented yet).
        """
        self.algorithm = algorithm
        self.coincidences_stats = coincidences_stats
        self.transition_memory = transition_memory
        self.requested_group_count = requested_group_count
        self.symmetrizeTAM = symmetrizeTAM
        self.purge_garbage = purge_garbage  # remove the biggest temporal group
        self.debug = debug

        self.TAM = None  # nxn matrix where n is the num. of  concidences
        self.PCG = None  # matrix
        self.temporal_groups = dict([])  # id => list of coincidence ids

        self.top_neighbors = 3  # maximum number of coincidences linked to the one with
                                # highest TC that will be processed
        self.group_max_size = 10  # maximum size of one temporal group
        self.temporal_group_count = 0

        self.sequence_buffer = None

    def learn(self, y, is_reset=False):
        """Learn temporal relations among concidenes.

        Args:
            y: the prob. distribution over coincidences given the evidence
            is_reset: if True resets the sequence buffer
        """
        if self.TAM is None:  # initialization
            # Time adjacency matrix
            self.TAM = np.zeros((self.coincidences_count,
                                 self.coincidences_count))
            if self.debug: print('TAM created with dimmensions', self.TAM.shape)

        if self.sequence_buffer is None:
            # creating buffer of the size of transition memory to temporally
            # store input patterns
            self.sequence_buffer = utils.RingBuffer(self.transition_memory + 1)

            self.winner_coincidences = []

        # reset signal is triggered also by an empty pattern
        if (y is not None and np.min(y) == np.max(y)) or is_reset:
            self.sequence_buffer.reset()
            return

        # id of the coincidence is the index of the maximal value in y
        # y is the vector of activations of coincidences, y[i] is proportional
        # to the distance between the evindence and the coincidence i
        input_id = np.argmax(y)
        self.winner_coincidences.append(input_id)

        self.sequence_buffer.append(input_id)

        # wait for the buffer to fill in or for the reset signal to update
        # the TAM
        if self.sequence_buffer.filled_items() < 2:
        # if len(self.sequence_buffer) < self.transition_memory/2:
        # if not self.sequence_buffer.is_filled():
            return

        #input_id = self.sequence_buffer[-1]  # last element

        # updating TAM
        # concidences that are closer in time get higher increment, the
        # increment linearly depends on the time proximity (the furthest gets
        # increment of one, the closest of transition_memory)
        for i, pattern_id in enumerate(self.sequence_buffer.get()):
            if pattern_id is not None and pattern_id is not input_id:
                self.TAM[pattern_id, input_id] += i + 1

        # reset buffer after updating TAM
        # if len(self.sequence_buffer) > 0:
        #     self.sequence_buffer.reset()

    def finalize_learning(self, grouping_method='AHC', spatial_pooler=None):
        """Finalize learning in the following steps:
         1. Remove rare coincidences (done in SpatialPooler)
         2. Compute coincidence priors
         3. Make T symmetric
         4. Normalize T by rows
         5. Temporal grouping
         6. Compute PCG
         """

        def add_to_temporal_group(c_id, g_id=None):
            """Add coincidence to a new or to an existing temporal group.

            Args:
                c_id: coincidence index
                g_id: existing temporal group index

            Returns:
                group id if creating a new one
            """
            if c_id not in nonassigned_coincidences:
                return
            nonassigned_coincidences.remove(c_id)
            if g_id is None:
                self.temporal_groups[len(self.temporal_groups)] = [c_id]
                return len(self.temporal_groups) - 1
            else:
                if (len(self.temporal_groups[g_id]) < self.group_max_size):
                    self.temporal_groups[g_id].append(c_id)

        # 2. Compute coincidence priors
        self.conincidence_prior = dict()
        count_sum = float(sum(self.coincidences_stats))
        for c_id, count in enumerate(self.coincidences_stats):
            self.conincidence_prior[c_id] = count / count_sum

        # visualize.show_image(np.asarray(self.conincidence_prior.values()).reshape(10,20))

        # 3. Make T symmetric
        if self.symmetrizeTAM:
            self.TAM = utils.symmetrize(self.TAM.T)

        # zero-out the diagonal
        for i in range(self.TAM.shape[0]):
            self.TAM[i, i] = 0

#         4. Normalize T by rows
#        for i in xrange(self.TAM.shape[0]):
#            for j in xrange(self.TAM.shape[1]):
#                if self.TAM[i].sum() > 0:
#                    self.TAM[i, j] /= float(self.TAM[i].sum())

        # normalize byt rows and columns
        row_max = self.TAM.max(axis=1).reshape((self.TAM.shape[1], 1))
        col_max = self.TAM.max(axis=0).reshape((self.TAM.shape[0], 1))
        self.TAM = np.nan_to_num(np.divide(self.TAM, np.sqrt(np.dot(row_max, col_max.T))))

#        visualize.show_matrix(self.TAM)

        # 5. Temporal grouping
        if grouping_method == "AHC":
            # AHC algorithm
            # http://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html
            # http://math.stanford.edu/~muellner/fastcluster.html
            import scipy.cluster.hierarchy as hier

            # AHC needs a distance matrix
            TAM_invs = 1 - self.TAM

#            Z = hier.average(TAM_invs)
            Z = hier.complete(TAM_invs)
            # Z = hier.weighted(TAM_invs)
            # Z = hier.centroid(TAM_invs)
            t = self.requested_group_count
            T = hier.fcluster(Z, t, criterion='maxclust')

            # T is a list of indices to groups for each of the coincidences
            # creating temporal groups based on T
            for c_id, g_id in enumerate(T):
                g_id = g_id - 1
                if not g_id in self.temporal_groups.keys():
                    self.temporal_groups[g_id] = [c_id]
                else:
                    self.temporal_groups[g_id].append(c_id)

        elif grouping_method == "Numenta":  # greedy algorithm
            nonassigned_coincidences = range(self.coincidences_stats)  # ids of coincidences

            while len(nonassigned_coincidences) > 0:
                # 5.1 Select the non-assigned coincidence c_i with the highest
                # temporal connection TC and add it to a new temporal group g_k.
                htc = -1  # highest temporal connection value
                htc_id = None  # id of the coincidence
                for i in nonassigned_coincidences:
                    if self.TAM[i].max() > htc:
                        htc = self.TAM[i].max()
                        htc_id = i
                assert(htc_id is not None)
                # add selected coincidence to a new temporal group
                g_id = add_to_temporal_group(htc_id)

                # 5.2 Pick at most topNeighbors non-assigned coincidences with the
                # highest temporal connection and pool them to the same group g_k
                j = 0
                tmp = dict()
                while len(self.temporal_groups[g_id]) < self.group_max_size and \
                        len(nonassigned_coincidences) > 0:
                    if not len(self.temporal_groups[g_id]) - 1 >= j:
                        break
                    htc_id = self.temporal_groups[g_id][j]
                    tmp.clear()
                    for k in range(self.TAM.shape[1]):
                        tmp[k] = self.TAM[htc_id, k]  # dict(c_id => temporal connection value)
                    del tmp[htc_id]  # remove previously selected c_id
                    sorted_tmp = sorted(tmp, key=itemgetter(1), reverse=True)[0:self.top_neighbors]
                    for c_id, tc in sorted_tmp:
                        add_to_temporal_group(c_id, g_id)
                    j += 1

        # 5.1 purge garbage group
        # garbage group is the largest one, TODO use better metric
        if self.purge_garbage and spatial_pooler is not None:
            # find the largest temporal group
            garbage_id = 0
            max_len = 0
            for g_id in self.temporal_groups.keys():
                if len(self.temporal_groups[g_id]) > max_len:
                    max_len = len(self.temporal_groups[g_id])
                    garbage_id = g_id

            # delete all the coincidences in that group
            spatial_pooler.coincidences = utils.multi_delete(spatial_pooler.coincidences, self.temporal_groups[garbage_id])
            self.coincidences_count = len(spatial_pooler.coincidences)
            spatial_pooler.coincidences_matrix = np.vstack(spatial_pooler.coincidences.values())

            # delete the temporal group and change the indices so that they are
            # continuous
            del self.temporal_groups[garbage_id]
            for i in range(garbage_id, len(self.temporal_groups)):
                self.temporal_groups[i] = self.temporal_groups[i + 1]
            del self.temporal_groups[len(self.temporal_groups) - 1]  # delete the last

            count = 0
            for g in self.temporal_groups.values():
                count += len(g)
            assert(count == self.coincidences_count)

        # 6. Compute PCG
        # 6.1
        self.PCG = np.zeros((self.coincidences_count, len(self.temporal_groups)))
        # for i in self.coincidences_stats.keys():
        for i in range(self.PCG.shape[0]):
            for j in range(self.PCG.shape[1]):
                if i in self.temporal_groups[j]:  # if c_i is in g_j
                    self.PCG[i, j] = self.conincidence_prior[i]  # assign P(c_i)
        # 6.2 each column in PCG should sum up to 1
        self.PCG = self.PCG.T
        for i in range(self.PCG.shape[0]):
            tsum = float(self.PCG[i].sum())
            if tsum > 0:
                self.PCG[i] /= tsum
        # self.PCG = self.PCG.T

    def infer(self, y, sparsify=False):
        """Compute the activations of temporal groups given the activations
        of coincidences from the temporal pooler.
        """
        if self.algorithm == 'maxProp':
            out = np.max(self.PCG * y, axis=1)  # max for each group (row)

        if self.algorithm == 'sumProp':
            out = np.dot(self.PCG, y)  # sum for each group (row)

        # TODO time-based inference
        if self.algorithm == 'tbi':
            pass

        if sparsify:
            # zero-out all but the maximal value
            max_index = np.argmax(out)
            out *= 0
            out[max_index] = 1

        return utils.normalize_to_one(out)
