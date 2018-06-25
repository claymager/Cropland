#!/bin/python
# -*- coding: utf-8 -*-
"""

File: helper_functions.py
Author: John Mager
Github: claymager
Description: grab-bag functions for use with these things

image_path is the path to an Operational Land Image directory (Landsat 8's image format):
    11 spectral band .TIF
    1 quality assesment band .TIF
    2 metadata files .TXT
represents one image

"""

import os
import string
from collections import Counter
import numpy as np
from osgeo import gdal
import matplotlib as mpl
import matplotlib.pyplot as plt


def get_metadata(image_path, query):
    """
    returns value of first matching metadata field for landsat MTL.txt files
    """
    mtl_filename = next(image_path + mtl for mtl in os.listdir(image_path) if "MTL" in mtl)
    for line in open(mtl_filename):
        if query in line:
            value = line.strip().split()[-1]
            return value
    return None


def generic_band(image_path):
    """ Gets a generic, formatable string for the bands in image_path
    takes directory path
    returns path with hole for specific bands
    """
    band3_name = next( b for b in os.listdir(image_path) if "B3" in b )
    generic_filename = band3_name.replace("B3", "B{}")
    generic_path = image_path + generic_filename
    return generic_path


def plot_crop_layer(cdl, cmap):
    """Plots a cdl with the full-scale of its cmap

    Args:
         cdl (np.array): TODO
         cmap (plt.cmap): TODO
    
    Raises:
        IndexError: if cdl is empty

    """
    temp0 = cdl[0,0]
    temp1 = cdl[-1,-1]
    cdl[0,0] = 0
    cdl[-1,-1] = 255
    plt.figure(dpi=400)
    plt.imshow(cdl, cmap=cmap)
    plt.xticks(())
    plt.yticks(())
    cdl[0,0] = temp0
    cdl[-1,-1] = temp1
    

#
# PURE FUNCTIONS
#


@np.vectorize
def make_mask(qa_code):
    """ Masks array from presence in list of acceptable conditions

    Args:
        qa_code (np.array): landsat-8 Quality Assessment data
    """
    return qa_code in [2720, 2724, 2728, 2732]

@np.vectorize
def adjust_gamma(value, gamma):
    inv_gamma = 1 / gamma
    return (value / 255) ** inv_gamma * 255


def get_gdal_colormap(geotif):
    """ Gets a matplotlib colormap from a geotif

    Args:
        geotif (gdal): A gdal image with 256 integer pixel values

    Returns: 
        a matplotlib cmap

    Raises:
        AttributeError: if geotif has no ColorTable

    """
    band = geotif.GetRasterBand(1)
    color_table = band.GetColorTable()
    num_colors = color_table.GetCount()
    colors = np.array([color_table.GetColorEntry(i) for i in range(num_colors)])
    colors = colors / num_colors
    color_map = mpl.colors.ListedColormap(colors)
    return color_map


def rank_labels(labels):
    """TODO: Docstring for rank_labels.

    Args:
        labels (TODO): TODO

    Returns: TODO

    """
    lbls = list(set(labels))
    if len(lbls) > 100:
        warn("Too many clusters: labels are not sorted")
        return
    if max(labels) > 99:
        lbl_dict = {value:index for index, value in enumerate(lbls)} 
        labels = [lbl_dict[l] for l in labels]
        
    labels = [string.printable[lbl] for lbl in labels]
    label_counts = Counter(labels)
    ranks = {lbl: rank for rank, (lbl, _) in
                enumerate(label_counts.most_common())}
    labels = np.array([ranks[lbl] for lbl in labels])
    return labels


def get_indexes( val, array ):
    return { index for index, label in enumerate(array) if label == val }


def pairwise_jaccard(left, right):
    """
    construct heatmap array of jaccard similarity scores for individual sets
    labels should be sequences starting at 0
    """
    assert len(left) == len(right)
    left_sets = { label: get_indexes(label, left) for label in set(left) }
    right_sets = { label: get_indexes(label, right) for label in set(right) }
    result_shape = (len(left_sets), len(right_sets))
    result = np.zeros(result_shape)
    for label_L, set_L in left_sets.items():
        for label_R, set_R in right_sets.items():
            intersection = len(set_L.intersection(set_R))
            union = len(set_L.union(set_R))
            result[label_L, label_R] = intersection / union
    return result

def plot_pairwise(input_array, true_labels, cmap="binary"):
    """ Plots an array produced by pairwise_jaccard by normalizing across rows

    Args:
        input_array (np.array): 2-dimensional array
        true_labels (list(str)): x_axis labels

    """
    row_normalized = np.array([ row / row.sum() for row in input_array ])
    fig = plt.figure(dpi=300, figsize=(12, 8))
    ax = fig.add_subplot(111)

    ax.imshow(row_normalized.T, cmap=cmap)
    plt.yticks(np.arange(input_array.shape[1]), true_labels)
    plt.xticks(np.arange(input_array.shape[0]))
    """ Normalizing across rows -> specific value is unimportant
    color_bar = fig.colorbar(ax.images[0])
    cbytick_obj = plt.getp(color_bar.ax.axes, "yticklabels")
    plt.setp(cbytick_obj, color="w", size=14)
    """
    ax.xaxis.label.set_color("white")
    ax.tick_params(axis="x", colors="white", labelsize=14)
    ax.tick_params(axis="y", colors="white", labelsize=14, size=6)

    

def most_common_labels( image, input_labels ):
    """ Returns list of labels used in image, ordered by prevalence.
    
    Args:
        image (np.array): array of integer codes
        input_labels (list[str]): 
    
    Returns:
        (list[str])
    
    TODO:    
        There's probably a redundancy with rank_labels
    
    """
    count = Counter(image.ravel())
    ranked_labels = []
    for code, _ in count.most_common():
        try:
            label = input_labels[code]
        except IndexError:
            warn("most_common_labels: Unknown label")
            label = "Unknown"
        ranked_labels.append(label)
    return ranked_labels
