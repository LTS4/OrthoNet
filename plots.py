#
# Authors: Mireille El Gheche, Giovanni Chierchia
#
# Date: July 2020
#
# This code is released under the CeCILL-B licence: https://spdx.org/licenses/CECILL-B.html
#

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


COLORS  = np.array(['#377eb8', '#ff7f00', '#4daf4a', '#a65628', '#f781bf', '#984ea3', '#999999', '#e41a1c', '#dede00'])
MARKERS = np.array(['o', '^', 's', 'X'])


def lighten_color(color_list, amount=0.25):
    import colorsys
    import matplotlib.colors as mc
    out = []
    for color in color_list:
        try:
            c = mc.cnames[color]
        except:
            c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        lc = colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
        out.append(lc)
    return out


def plot_clustering(X, y, ax=None):
    if not ax: ax = plt.gca()
    ec  = COLORS[y%len(COLORS)]
    for i, mi in zip(np.unique(y), MARKERS):
        mask = i == y
        ax.scatter(X[mask,0], X[mask,1], s=40, c=lighten_color(ec[mask]), edgecolors=ec[mask], alpha=0.8, linewidths=2, marker=mi)
    ax.set_xticks(())
    ax.set_yticks(())
    
    
def plot_graph(X, y, affinity, ax):  
    pos, ypos = X.copy(), y.copy()
    
    non_zero = np.triu(affinity, k=1) > 0
    sources, targets = np.where(non_zero)
    new_pos = np.zeros_like(pos)
    for i in np.unique(y):
        new_pos[y==i] = pos[ypos==i]
    
    segments = np.stack((new_pos[sources, :], new_pos[targets, :]), axis=1)
    lc = LineCollection(segments, zorder=0, colors=(0.3,0.3,0.3),  alpha=0.9, linewidth=1)
    lc.set_linewidths(1)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_xlim(segments[:,:,0].min(), segments[:,:,0].max())
    ax.set_ylim(segments[:,:,1].min(), segments[:,:,1].max())
    ax.add_collection(lc)
    ax.scatter(new_pos[:, 0], new_pos[:, 1], s=20, c='w', edgecolors='k', alpha=0.9)
    
    
def show_datasets(sets, y_preds=None):
    plt.figure(figsize=(len(sets) * 3 + 3, 3))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)
    for i, set_name in enumerate(sets):
        s = sets[set_name]
        X = s["X"]
        y = s["y"]
        plt.subplot(1, len(sets), i+1)
        if y_preds==None:
            plot_clustering(X, y)
        else:
            plot_clustering(X, y_preds[i])
    plt.show()
    
    
def show_graphs(sets):
    plt.figure(figsize=(len(sets) * 3 + 3, 3))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)
    for i, set_name in enumerate(sets):
        s = sets[set_name]
        X = s["X"]
        y = s["y"]
        G = s["G"]
        ax = plt.subplot(1, len(sets), i+1)
        plot_graph(X, y, G, ax)
        

def show_results(results, sets, experiment_name, save=False):
    n = len(sets)
    fig, ax = plt.subplots(n, 5, figsize=(18,2.5*n))
    plt.subplots_adjust(hspace=.05)
    for i, set_name in enumerate(sets):   
        plot_results(results, sets, experiment_name, set_name, ax[i])
    plt.setp(ax, xticks=[], yticks=[])
    plt.show()
    if save:
        for i in range(ax.shape[1]):
            extent0 = ax[0][i].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            extent1 = ax[1][i].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            points0 = extent0.get_points()
            points1 = extent1.get_points()
            points0[0,1] = points1[0,1]
            extent  = matplotlib.transforms.Bbox(points0)
            fig.savefig(experiment_name+'_'+str(i)+'.png', dpi=200, pad_inches=0, bbox_inches=extent)#='tight')


def plot_results(results, sets, experiment_name, set_name, ax):
    
    X = sets[set_name]["X"]
    y = sets[set_name]["y"]
    K = sets[set_name]['K']
    G = sets[set_name]['G']
    E = results[experiment_name][set_name]['embedding']
    
    clf = KMeans(K, random_state=42)
    P = clf.fit_predict(E)
    C = clf.cluster_centers_
    i = clf.predict(C)
    
    if E.shape[1] > 2:
        sne = TSNE(n_components=2, random_state=42)
        TT = sne.fit_transform( np.concatenate([E,C]) )
        E = TT[:-C.shape[0]]
        C = TT[-C.shape[0]:]
    
    ax[0].scatter(X[:,0], X[:,1], s=40, c=lighten_color('k', 0.1), edgecolors='k', alpha=0.8)
    plot_graph(X, y, G, ax[1])
    ax[2].scatter(E[:,0], E[:,1], s=40, c=lighten_color('k', 0.1), edgecolors='k', alpha=0.8)
    ax[3].scatter(E[:,0], E[:,1], s=40, c=lighten_color(COLORS[P]), edgecolors=COLORS[P], alpha=0.8)
    ax[3].scatter(C[:,0], C[:,1], s=400, c=COLORS[i], edgecolors='k', marker='*', alpha=0.8)
    #ax[3].scatter(X[:,0], X[:,1], s=40, c='w', edgecolors=COLORS[P], alpha=0.8)
    plot_clustering(X, P, ax[4])
    
    net = results[experiment_name][set_name]['net']
    if net:
        GX, GY = build_grid(X, num=300)
        XY = np.stack([GX.flat,GY.flat],axis=1)
        GE = net.predict(XY)
        GP = clf.predict(GE).reshape(GX.shape)
        ax[4].contour(GX, GY, GP, colors='k', linewidths=1)#, levels=np.unique(y))
            
            
def build_grid(E, num=100):
    vmin = E.min(axis=0)
    vmax = E.max(axis=0)
    vmin -= 0.01
    vmax += 0.01
    xrange = np.linspace(vmin[0],vmax[0], num)
    yrange = np.linspace(vmin[1],vmax[1], num)
    x, y = np.meshgrid(xrange, yrange)
    return x, y
        