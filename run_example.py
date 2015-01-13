#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import os.path
import numpy as np

import pyhtm.htm as htm
from pyhtm.nodes.imagesensor import ImageSensor
from pyhtm.utils import extract_patches

from sklearn.neighbors import KNeighborsClassifier


#
# == Configuration ===
#
dataset = {
    'name': 'Pictures: subset',
    'path': 'pictures-subset',
    'image_width': 32,
    'image_height': 32,
    'image_background': 1,  # white
    'image_mode': 'bw'
}

train_data = os.path.join('data', dataset['path'], 'train')
test_data = os.path.join('data', dataset['path'], 'test')

show_graphs = True
debug = True
verbose = True

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
        'size': [8, 8],

    # Spatial pooler
        'maxCoincidenceCount': 200,
        'spatialPoolerAlgorithm': 'gaussian',
        'sigma': 1.2,
        'maxDistance': 0.01,

    # Temporal pooler
        'requestedGroupsCount': 30,
        'temporalPoolerAlgorithm': 'maxProp',
        'transitionMemory': 4,
        'symmetrizeTAM': True
    },

    # Level 1
    {
        'nodeCloning': True,
        'size': [4, 4],

    # Spatial pooler
        'maxCoincidenceCount': 300,
        'spatialPoolerAlgorithm': 'product',  # dot
        'sigma': 1.2,
        'maxDistance': 0.001,

    # Temporal pooler
        'requestedGroupsCount': 40,
        'temporalPoolerAlgorithm': 'maxProp',
        'transitionMemory': 8,
    }
    ]

learning_params = [
    # Level 0
    {'sp':
        # Spatial pooler
        {
        # TODO bounding box problem!
        'explorer': ['RandomSweep', {'sweepOffObject':False,
                                     'sweepDirections': 'all'}],
        # 'explorer': ['RandomJump', {'jumpOffObject': False}],
        'numIterations': 300
        },

    'tp':
        # Temporal pooler
        {
        'explorer': ['RandomSweep', {'sweepOffObject': False,
                                     'sweepDirections': 'all'}],
        # 'explorer': ['ExhaustiveSweep', {'sweepOffObject':False}],
        'numIterations': 20000
        },
    },

    # Level 1
    {'sp':
        {
        'explorer': ['RandomSweep', {'sweepOffObject':False,
                                     'sweepDirections': 'all'}],
        # 'explorer': ['RandomJump', {'jumpOffObject': False}],
        'numIterations': 300
        },

    'tp':
        # Level 1: # Temporal pooler
        {
        'explorer': ['RandomSweep', {'sweepOffObject':False,
                                     'sweepDirections': 'all'}],
        'numIterations': 20000
        },
    }
]

# assuming square image and level size
patch_size = [sensor_params['width'] / net_params[0]['size'][0]] * 2

# creating sensor
sensor = ImageSensor(width=sensor_params['width'],
                     height=sensor_params['height'],
                     background=sensor_params['background'],
                     mode=sensor_params['mode'])
sensor.loadMultipleImages(train_data)

# create the network
htmNet = htm.HTM(sensor, debug=debug, show_graphs=show_graphs, verbose=verbose)
htmNet.create_network(net_params)

#
# == Learning ==
#

htmNet.learn(learning_params)

#
# == Testing ===
#
print('Testing...')
start_time = time.clock()

# getting training data
sensor.loadMultipleImages(train_data)
sensor.setParameter('explorer', 'Flash')
print('  Num of train images: %d' % sensor.getNumIterations())

train = []  # infered output of HTM for each of the original images
train_set_orig = []  # original training images
train_labels = []  # labels of training original images
for i in range(sensor.getNumIterations()):
    data = sensor.compute()
    im, cat = data['data'], data['category']
    train_set_orig.append(im.reshape(np.prod(im.shape), ))
    train_labels.append(cat)
    out = htmNet.infer(extract_patches(im, patch_size), merge_output=True)
    train.append(np.asarray(out))

# getting testing data
sensor.clearImageList()
sensor.loadMultipleImages(test_data)
sensor.setParameter('explorer', 'Flash')
print('  Num of test images: %d' % sensor.getNumIterations())

test = []  # infered output of HTM for each of the original images
test_set_orig = []  # original testing images
test_labels = []  # labels oftesting original images
for i in range(sensor.getNumIterations()):
    data = sensor.compute()
    im, cat = data['data'], data['category']
    test_set_orig.append(im.reshape(np.prod(im.shape), ))
    test_labels.append(cat)
    out = htmNet.infer(extract_patches(im, patch_size), merge_output=True)
    test.append(np.asarray(out))


# kNN in HTM space
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(train, train_labels)
test_score = knn.score(test, test_labels)

# kNN in original space
knn_orig = KNeighborsClassifier(n_neighbors=1)
knn_orig.fit(train_set_orig, train_labels)
test_score_orig = knn_orig.score(test_set_orig, test_labels)

print('The test code ran for %.2fm' % ((time.clock() - start_time) / 60.))

print('Testing has completed with classification accuracy of %.4f %%. CA in original space is %.4f %%.'
       % (test_score * 100., test_score_orig * 100.))
