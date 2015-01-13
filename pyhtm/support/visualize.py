import matplotlib.pyplot as plt
import matplotlib.cm as cm
from math import ceil


def show_temporal_groups(node, patch_size=[8, 8]):
    """Print and plots temporal groups of a node.
    """

    max_width = max(len(x) for x in node.tp.temporal_groups.values())
    i = 0
    for g in node.tp.temporal_groups.values():
        for c_id in g:
            i += 1
            plt.subplot(len(node.tp.temporal_groups), max_width, i)
            plt.imshow(node.sp.coincidences[c_id].reshape(patch_size),
                       cmap=cm.Greys_r, vmin=0, vmax=1)
            plt.axis('off')
        if i % max_width != 0:  # start new row
            i += max_width - (i % max_width)
    plt.show()


def show_codebook(node, patch_size=[8, 8]):
    rows = ceil(len(node.sp.coincidences) ** .5)
    cols = ceil(len(node.sp.coincidences) / rows) + 1
    for i, c in node.sp.coincidences.items():
        plt.subplot(rows, cols, i + 1)  # TODO make universal
        plt.imshow(c.reshape(patch_size), cmap=cm.Greys_r, vmin=0, vmax=1)
        plt.axis('off')
    plt.show()
#    print node.sp.coincidences_stats


def show_matrix(m):
    plt.imshow(m)
    plt.show()


def show_TAM(node):
    show_matrix(node.tp.TAM)


def show_sorted_TAM(node):
    """Show sorted TAM. The TAM is sorted so that neighboring cols and rows
    represent coincidences from the same temporal group.
    """
    import numpy as np

    sorted_TAM = np.zeros(node.tp.TAM.shape)

    i = 0
    for g in node.tp.temporal_groups.values():
        for c_id in g:
            sorted_TAM[i, :] = node.tp.TAM[c_id, :]
            sorted_TAM[:, i] = node.tp.TAM[:, c_id]
            i = i + 1

    shape = (sorted_TAM.shape[0],
             sorted_TAM.shape[1] + len(node.tp.temporal_groups))
    sorted_TAM_delim = np.ones(shape)

    # show delimiters
    i = 0
    j = 0
    for g in node.tp.temporal_groups.values():
        for c_id in g:
            sorted_TAM_delim[:, i] = sorted_TAM[:, j]
            i = i + 1
            j = j + 1
        i = i + 1

    show_matrix(sorted_TAM_delim)


def show_PCG(node):
    show_matrix(node.tp.PCG)


def show_image(im):
    plt.imshow(im, cmap=cm.Greys_r, vmin=0, vmax=1)
    plt.show()


def show_images(imgs):
    for i, im in enumerate(imgs):
        plt.figure(i)
        plt.imshow(im, cmap=cm.Greys_r, vmin=0, vmax=1)
    plt.show()
