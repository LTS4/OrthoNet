#
# Authors: Mireille El Gheche, Giovanni Chierchia
#
# Date: July 2020
#
# This code is released under the CeCILL-B licence: https://spdx.org/licenses/CECILL-B.html
#

from collections import OrderedDict

import numpy as np

import scipy
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree

from sklearn import datasets
from sklearn.neighbors import kneighbors_graph


def generate_data():
    n_samples = 400
    sets = OrderedDict()

    X, y = datasets.make_circles(n_samples, factor=.5, noise=.05)
    G = build_graph(X)
    sets['circles'] = {'X': X, 'y': y, 'K': 2, 'G': G}

    X, y = datasets.make_moons(2 * n_samples // 3, noise=.05, random_state=42)
    X = np.concatenate((X[y==1], X[y==0]-(0.3,-0.1), X[y==0]+(2.3,0.1)))
    y = np.concatenate((y[y==1], y[y==0], y[y==0]+2))   
    X = (X - X.mean(0)) / X.std(0)
    G = build_graph(X)
    sets['moons'] = {'X': X, 'y': y, 'K': 3, 'G': G}

    X, y = datasets.make_moons(n_samples//2, noise=.05, random_state=42)
    X = np.concatenate((X[y==1], X[y==0]-(0.3,-0.1)))
    y = np.concatenate((y[y==1], y[y==0]))
    X = np.concatenate((X,X+(2.65,0.1)))
    y = np.concatenate((y,y+2))   
    G = build_graph(X)
    sets['hills'] = {'X': X, 'y': y, 'K': 4, 'G': G}
    
    return sets


def build_graph(X, neighbors=8, mst_weight=10):
    A = kneighbors_graph(X, neighbors, 'distance').toarray()
    D = squareform(pdist(X))
    MST = minimum_spanning_tree(D).toarray()
    MST *= mst_weight
    A = np.maximum(A, MST)
    A = (A + A.T) / 2
    A[A.nonzero()] = np.exp(-A[A.nonzero()])
    return A

