#!/usr/bin/env python

import os
import numpy as np
import unittest

import pyhtm.nodes.poolers as poolers
import pyhtm.utils as utils

patterns = np.asarray([
    [1,0,1,1],
    [0,0,1,0],
    [0,0,1,1],
    [1,0,1,1],
    [1,0,0,1],
])
num_of_unique_patterns = 4

class TestSpatialPoolerGaussian(unittest.TestCase):

    def runTest(self):
        sp = poolers.SpatialPooler(algorithm='gaussian',
                                   ignore_background_pattern=False)

        for p in patterns:
            sp.learn(p)
        sp.finalize_learning()

        self.assertEqual(len(sp.coincidences), 4)

        # the infered vector sums to 1 (it's a probability distribution)
        pattern = np.random.random((10, 1))
        y = sp.infer(pattern[0])
        self.assertAlmostEqual(y.sum(), 1.0)

        # numeric test
        y = sp.infer(patterns[0])
        self.assertAlmostEqual(y[0], 0.77580349)

class TestSpatialPoolerDot(unittest.TestCase):

    def runTest(self):
        sp = poolers.SpatialPooler(algorithm='dot',
                                   ignore_background_pattern=False)

        sp.learn(patterns[0])
        sp.learn(patterns[1])
        sp.finalize_learning()

        self.assertEqual(len(sp.coincidences), 2)
        self.assertTrue(np.array_equal(sp.coincidences[0], patterns[0]))

        y = sp.infer(patterns[0])
        self.assertEqual(y[0], 0.75)

        y = sp.infer(patterns[2])
        self.assertAlmostEqual(y[0], 0.66666666)

class TestSpatialPoolerProduct(unittest.TestCase):

    def testAlgorithm(self):
        sp = poolers.SpatialPooler(algorithm='product',
                                   ignore_background_pattern=False)
        evidence = [0.8, 0.1,  # from node 0
                    0.1, 0.5,  # from node 1
                    0.2, 0.3]  # from node 2
        coincidence = [0, 0, 1]  # 0.8 * 0.1 * 0.3

        product = sp._product_inference(coincidence, evidence)
        self.assertAlmostEqual(product, 0.024)


        sp = poolers.SpatialPooler(algorithm='product',
                                   ignore_background_pattern=False)
        pp = patterns.copy()
        sp.learn(pp)
        sp.finalize_learning()

        coincidence = np.array([0, 2, 2, 0, 0])
        evidence = np.array([0.6, 0.1, 0.2, 0.1,
                             0, 0.1, 0.8, 0.1,
                             0, 0, 0.4, 0.6,
                             0.8, 0, 0.2, 0,
                             0.5, 0, 0.4, 0.1])
        y = float(sp._product_inference(coincidence, evidence))
        # 0.6 * 0.8 * 0.4 * 0.8 * 0.5
        self.assertAlmostEqual(y, 0.0768)

    def testDistance(self):
        sp = poolers.SpatialPooler()

        x, y = np.asarray([1, 4, 2, 3, 1, 0]), np.asarray([1, 0, 2, 3, 1, 2])
        dist = sp._widx_distance(x, y)
        self.assertEqual(dist, 2)

        y = np.asarray([1, 4, 2, 3, 1, 0])
        dist = sp._widx_distance(x, y)
        self.assertEqual(dist, 0)

        x, y = np.asarray([0.1, 0.55, 0.31]), np.asarray([0.2, 0.55, 0.31])
        dist = sp._widx_distance(x, y)
        self.assertEqual(dist, 1)

    def testFromChild(self):
        sp = poolers.SpatialPooler(algorithm='product',
                                   ignore_background_pattern=False)

#        correct_coincidence = np.array([1, 0, 0, 0,
#                                        0, 0, 1, 0,
#                                        0, 0, 1, 0,
#                                        1, 0, 0, 0,
#                                        1, 0, 0, 0])
        correct_coincidence = np.array([0, 2, 2, 0, 0])

        # patterns are now considered to be the outputs from children nodes
        # so only one concatenated pattern of all the patterns in the array
        # will be stored in the codebook
        pp = patterns.copy()
        sp.learn(pp)
        sp.finalize_learning()

        self.assertEqual(len(sp.coincidences), 1)
        # the sum of ones in the concidence is equal to the number of children
        self.assertEqual(np.sum(sp.coincidences[0]), num_of_unique_patterns)
        self.assertTrue(np.array_equal(sp.coincidences[0], correct_coincidence))

        pattern = np.random.random((20, 1))
        y = sp.infer(pattern)
        # there is only one coincidence, so the normalized output must be 1
        self.assertEqual(y, [1.0])

class TestTemporalPooler(unittest.TestCase):

    def runTest(self):
        sp = poolers.SpatialPooler(algorithm='gaussian', max_distance=0,
                rare_coincidence_threshold=0, ignore_background_pattern=False,
                debug=False)

        for p in patterns:
            sp.learn(p)

        sp.finalize_learning()
        self.assertEqual(len(sp.coincidences), num_of_unique_patterns)

        tp = poolers.TemporalPooler(coincidences_stats=sp.coincidences_stats,
                                    transition_memory=2, algorithm='maxProp',
                                    requested_group_count=4)
        tp.coincidences_count = len(sp.coincidences)

        for p in patterns:
            tp.learn(sp.infer(p))

        desired_TAM = np.array([[ 2.,  4.,  3.,  5.],
                                [ 3.,  0.,  4.,  2.],
                                [ 4.,  0.,  0.,  3.],
                                [ 0.,  0.,  0.,  0.]])
#        self.assertTrue(np.array_equal(tp.TAM, desired_TAM))

        tp.finalize_learning(grouping_method='AHC')

        # print tp.temporal_groups
        out = tp.infer(patterns[0])
        self.assertTrue(np.array_equal([0.2, 0.4, 0.4], out))

from pyhtm.nodes import BaseNode

class TestBaseNode(unittest.TestCase):

    def runTest(self):
        params = {'transitionMemory': 4, 'requestedGroupsCount': 2,
                  'ignoreBackgroundPattern': False}
        n = BaseNode(params)

        for p in patterns:
            n.do_spatial_learning(p)

        for p in patterns:
            n.do_temporal_learning(p)

        desired_TAM = np.array([[ 2.,  4.,  3.,  5.],
                                [ 3.,  0.,  4.,  2.],
                                [ 4.,  0.,  0.,  3.],
                                [ 0.,  0.,  0.,  0.]])
        self.assertTrue(np.array_equal(n.tp.TAM, desired_TAM))

from pyhtm.level import HTMLevel

class TestLevel(unittest.TestCase):

    def testCreatingLinks(self):
        l = HTMLevel(0, {'size': [1], 'nodeCloning': True}, previous_level_size=[1])
        prev_size = [8, 8]
        correct_links = {0: range(np.prod(prev_size))}

        links = l._get_links([1, 1], prev_size)

        self.assertEqual(links, correct_links)

    def runTest(self):
        params = {'transitionMemory': 4, 'requestedGroupsCount': 3}
        l = HTMLevel(0, {'size': [1], 'nodeCloning': True}, previous_level_size=[1])

        for p in patterns:
            l.do_spatial_learning([p])

        for p in patterns:
            l.do_temporal_learning([p])

        desired_TAM = np.array([[ 2.,  4.,  3.,  5.],
                                [ 3.,  0.,  4.,  2.],
                                [ 4.,  0.,  0.,  3.],
                                [ 0.,  0.,  0.,  0.]])
        self.assertEqual(np.array_equal(l.nodes[0].tp.TAM, desired_TAM), True)

        l.finalize_learning()
        self.assertEqual(len(l.nodes[0].tp.temporal_groups),
                                params['requestedGroupsCount'])


        # testing creating nodes using params
        params = {
            'nodeCloning': True,
            'size': [8, 8],
            'overlap': [0, 0],

            # Spatial pooler
                'maxCoincidenceCount': 128,
                'spatialPoolerAlgorithm': 'gaussian',
                'sigma': 1,
                'maxDistance': 0.1,

            # Temporal pooler
                'requestedGroupsCount': 20,
                'temporalPoolerAlgorithm': 'maxProp',
                'transitionMemory': 4,
            }
        l = HTMLevel(1, params, previous_level_size=[1])
        self.assertEqual(len(l.nodes), 1)

from pyhtm.htm import HTM
from pyhtm.nodes.imagesensor import ImageSensor

class TestHTM(unittest.TestCase):

    def testMultipleLevelInference(self):
        return
        from datasets import DatasetConfig

        dataset = DatasetConfig().load('pictures-subset')

        train_data = dataset['test_data_path']#dataset['train_data_path']
        test_data = dataset['test_data_path']

        sensor_params = {
            'width': dataset['image_width'],
            'height': dataset['image_height'],
            'background': dataset['image_background'],
            'mode': dataset['image_mode']
        }

        net_params = [
        # Level 0
        {
            'nodeCloning': True,
            'size': [4, 4],
            'overlap': [0, 0],

            # Spatial pooler
                'maxCoincidenceCount': 64,
                'spatialPoolerAlgorithm': 'gaussian',
                'sigma': 1,
                'maxDistance': 0.1,

            # Temporal pooler
                'requestedGroupsCount': 20,
                'temporalPoolerAlgorithm': 'sumProp',
                'transitionMemory': 4,
        },
        # Level 1
        {
            'nodeCloning': True,
            'size': [2, 2],
            'overlap': [0, 0],

            # Spatial pooler
                'maxCoincidenceCount': 128,
                'spatialPoolerAlgorithm': 'product',
                'sigma': 1,
                'maxDistance': 0.1,

            # Temporal pooler
                'requestedGroupsCount': 10,
                'temporalPoolerAlgorithm': 'sumProp',
                'transitionMemory': 8,
        },
        # Level 2
        {
            'nodeCloning': True,
            'size': [1],
            'overlap': [0, 0],

            # Spatial pooler
                'maxCoincidenceCount': 128,
                'spatialPoolerAlgorithm': 'product',
                'sigma': 1,
                'maxDistance': 0.1,

            # Temporal pooler
                'requestedGroupsCount': 20,
                'temporalPoolerAlgorithm': 'sumProp',
                'transitionMemory': 10,
        },
        # Level 3
        # {
        #     'nodeCloning': True,
        #     'size': [1],
        #     'overlap': [0, 0],

        #     # Spatial pooler
        #         'maxCoincidenceCount': 128,
        #         'spatialPoolerAlgorithm': 'gaussian',
        #         'sigma': 1,
        #         'maxDistance': 0.1,

        #     # Temporal pooler
        #         'requestedGroupsCount': 30,
        #         'temporalPoolerAlgorithm': 'maxProp',
        #         'transitionMemory': 10,
        # }
        ]

        learning_params = [
            # Level 0
            {'sp':
                # Spatial pooler
                {
                'explorer': ['RandomSweep', {'sweepOffObject':False,
                                             'sweepDirections': 'all'}],
                'numIterations': 300
                },

            'tp':
                # Temporal pooler
                {
                'explorer': ['RandomSweep', {'sweepOffObject': False,
                                             'sweepDirections': 'all'}],
                'numIterations': 10000
                },
            },
            # Level 1
            {'sp':
                # Spatial pooler
                {
                'explorer': ['RandomSweep', {'sweepOffObject':False,
                                             'sweepDirections': 'all'}],
                'numIterations': 200
                },

            'tp':
                # Temporal pooler
                {
                'explorer': ['RandomSweep', {'sweepOffObject': False,
                                             'sweepDirections': 'all'}],
                'numIterations': 5000
                },
            },
            # Level 2
            {'sp':
                # Spatial pooler
                {
                'explorer': ['RandomSweep', {'sweepOffObject':False,
                                             'sweepDirections': 'all'}],
                'numIterations': 200
                },

            'tp':
                # Temporal pooler
                {
                'explorer': ['RandomSweep', {'sweepOffObject': False,
                                             'sweepDirections': 'all'}],
                'numIterations': 5000
                },
            },
             # Level 3
            {'sp':
                # Spatial pooler
                {
                'explorer': ['RandomSweep', {'sweepOffObject':False,
                                             'sweepDirections': 'all'}],
                'numIterations': 200
                },

            'tp':
                # Temporal pooler
                {
                'explorer': ['RandomSweep', {'sweepOffObject': False,
                                             'sweepDirections': 'all'}],
                'numIterations': 5000
                },
            }
        ]
        train_data = 'data/pictures-subset/train'
        test_data = 'data/pictures-subset/test'

        sensor = ImageSensor(width=sensor_params['width'],
                             height=sensor_params['height'],
                             background=sensor_params['background'],
                             mode=sensor_params['mode'])
        net = HTM(sensor, verbose=False)
        net.create_network(net_params)
        sensor.loadMultipleImages(train_data)

        net.learn(learning_params)

        # getting testing data
        sensor.clearImageList()
        sensor.loadMultipleImages(test_data)
        sensor.setParameter('explorer', 'Flash')

        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        for i in range(40):
            data = sensor.compute()
            im, cat = data['data'], data['category']
            print(cat)
            patterns = utils.extract_patches(im, net.patch_size)
            if cat == 0:
                plt.plot(net.infer(patterns), color='b')
            else:
                plt.plot(net.infer(patterns), color='r')

        plt.show()
        # print net.infer(patterns)

    def testSaveAndLoad(self):
        sensor_params = {
            'width': 32,
            'height': 32,
            'background': 1,
            'mode': 'bw'
        }
        net_params = [{
            'nodeCloning': True,
            'size': [2, 2],
            'overlap': [0, 0],

            # Spatial pooler
                'maxCoincidenceCount': 128,
                'spatialPoolerAlgorithm': 'gaussian',
                'sigma': 1,
                'maxDistance': 0.1,

            # Temporal pooler
                'requestedGroupsCount': 20,
                'temporalPoolerAlgorithm': 'maxProp',
                'transitionMemory': 4,
        }]

        learning_params = [
            # Level 0
            {'sp':
                # Spatial pooler
                {
                'explorer': ['RandomSweep', {'sweepOffObject':False,
                                             'sweepDirections': 'all'}],
                'numIterations': 100
                },

            'tp':
                # Temporal pooler
                {
                'explorer': ['RandomSweep', {'sweepOffObject': False,
                                             'sweepDirections': 'all'}],
                'numIterations': 1000
                },
            },
        ]
        train_data = 'data/pictures-subset/train'

        sensor = ImageSensor(width=sensor_params['width'],
                             height=sensor_params['height'],
                             background=sensor_params['background'],
                             mode=sensor_params['mode'])
        net = HTM(sensor, verbose=False)
        net.create_network(net_params)
        sensor.loadMultipleImages(train_data)

        net.learn(learning_params)
        net_filename = 'network.pkl'
        net.save(filename=net_filename)

        self.assertTrue(os.path.exists(net_filename))
        newNet = HTM()
        newNet.load(filename=net_filename)
        os.unlink(net_filename)

        newNet.sensor.clearImageList()
        newNet.sensor.setParameter('explorer', 'Flash')

        self.assertListEqual(net.patch_size, newNet.patch_size)

    def testSegment(self):
        # TODO
        return True
        sensor_params = {
            'width': 32,
            'height': 32,
            'background': 1,
            'mode': 'bw'
        }
        net_params = [{
            'nodeCloning': True,
            'size': [2, 2],
            'overlap': [0, 0],

            # Spatial pooler
                'maxCoincidenceCount': 128,
                'spatialPoolerAlgorithm': 'gaussian',
                'sigma': 1,
                'maxDistance': 0.1,

            # Temporal pooler
                'requestedGroupsCount': 20,
                'temporalPoolerAlgorithm': 'maxProp',
                'transitionMemory': 4,
        }]

        learning_params = [
            # Level 0
            {'sp':
                # Spatial pooler
                {
                'explorer': ['RandomSweep', {'sweepOffObject':False,
                                             'sweepDirections': 'all'}],
                'numIterations': 100
                },

            'tp':
                # Temporal pooler
                {
                'explorer': ['RandomSweep', {'sweepOffObject': False,
                                             'sweepDirections': 'all'}],
                'numIterations': 1000
                },
            },
        ]
        train_data = 'data/pictures-subset/train'

        sensor = ImageSensor(width=sensor_params['width'],
                             height=sensor_params['height'],
                             background=sensor_params['background'],
                             mode=sensor_params['mode'])
        net = HTM(sensor, verbose=False)
        net.create_network(net_params)
        sensor.loadMultipleImages(train_data)

        net.learn(learning_params)

        sensor.clearImageList()
        sensor.loadSingleImage('data/test_clean.png')

        data = sensor.compute()
        im_clean, cat = data['data'], data['category']
        patch_size = net.patch_size

        im_clean_fw = net.infer(utils.extract_patches(im_clean, patch_size), merge_output=True)

        weights = np.asarray(net.segment(im_clean_fw))
#        print weights


class TestUtils(unittest.TestCase):

    def testNormalizeToOne(self):
        data = ((1, 0, 0.0), (0.5, 4, 3), (0.1, 0.1, 0.1))
        data = np.asarray(data)

        for x in data:
            output = utils.normalize_to_one(x)
            self.assertEqual(output.sum(), 1)

        bad_data = ((0, 0, 0), (2, 0, -2))
        bad_data = np.asarray(bad_data)

        for x in bad_data:
            self.assertRaises(ValueError, utils.normalize_to_one, x)

    def testNested(self):
        nested_list1 = [[1, 2, 3], [4, 5, 6]]
        nested_list2 = [[1,0,1,1],
                        [0,0,1,0],
                        [0,0,1,1],
                        [1,0,1,1],
                        [1,0,0,1]]

        self.assertTrue(utils.is_nested_list(nested_list1))
        self.assertTrue(utils.is_nested_list(nested_list2))

        self.assertFalse(utils.is_nested_list([]))
        self.assertFalse(utils.is_nested_list([0]))
        self.assertFalse(utils.is_nested_list([0, 1, 2]))
        self.assertFalse(utils.is_nested_list(0))

    def testFlatten(self):
        nested_list = [[1, 2, 3], [4, 5, 6]]
        correct_list = [1, 2, 3, 4, 5, 6]

        flatten_list = utils.flatten_list(nested_list)

        self.assertEqual(len(flatten_list), 6)
        self.assertListEqual(flatten_list, correct_list)

    def testSymmetrize(self):
        a = np.array([[1, 2, 3], [0, 1, 1], [1, 2, 3]])
        a = utils.symmetrize(a)

        self.assertTrue(np.all(a == a.T))

    def testFlattenPatches(self):
        from PIL import Image

        image = np.asarray(Image.open('./tests/test.jpg').convert('L').resize((64, 64)))
        image_size = image.shape
        patch_size = (4, 4)
        patches = utils.extract_patches(image, patch_size)

        im = utils.flatten_patches(patches, patch_size, image_size)

        self.assertEqual(im.shape[0], image_size[0]);
        self.assertEqual(im.shape[1], image_size[1]);

    def testCount_patches(self):
        data = [
            {'im_size': (100, 100), 'patch_size': (5, 5), 'overlap_size': (0, 0),
             'result': (20, 20)},
            {'im_size': (100, 200), 'patch_size': (5, 5), 'overlap_size': (0, 0),
             'result': (20, 40)},

            {'im_size': (32, 32), 'patch_size': (4, 4), 'overlap_size': (2, 2),
             'result': (15, 15)},
            {'im_size': (16, 16), 'patch_size': (4, 4), 'overlap_size': (2, 2),
             'result': (7, 7)},
        ]
        for d in data:
            result = utils.count_patches(
                d['im_size'], d['patch_size'], d['overlap_size']
            )
            self.assertTupleEqual(result, d['result'])

    def testExtract_patches(self):
        from PIL import Image

        image = Image.open('./tests/test.jpg').convert('L').resize((16, 16))
        image_arr = np.asarray(image)

        patch_size = (4, 4)
        overlap = (2, 2)
        patches = utils.extract_patches(image_arr, patch_size, overlap)
        self.assertEqual(len(patches), 49)

        patch_size = (8, 8)
        # overlap = (2, 2)
        patches = utils.extract_patches(image_arr, patch_size, overlap)
        self.assertEqual(len(patches), 4)

        # patch_size = (8, 8)
        overlap = (0, 0)
        patches = utils.extract_patches(image_arr, patch_size, overlap)
        self.assertEqual(len(patches), 4)

        image_arr = np.asarray(image.resize((20, 20)))
        # patch_size = (8, 8)
        overlap = (2, 2)
        patches = utils.extract_patches(image_arr, patch_size, overlap)
        self.assertEqual(len(patches), 9)

    def testMulti_delete(self):
        l = [1, 2, 3, 4, 5]
        l = utils.multi_delete(l, [1, 3])

        self.assertListEqual(l, [1, 3, 5])

    def testMaxIndices(self):
        l = [[23, 1, 4], [2.4, 5, 11], [44, 22, 105, 18]]
        r = utils.get_max_indices(l)
        self.assertTrue(np.array_equal(r, np.array([0, 2, 2])))


if __name__ == '__main__':
    unittest.main()
