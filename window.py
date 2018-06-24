#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

File: window.py
Author: John Mager
Github: claymager
Description: contains classes for landsat analysis
"""


import os
import string
from warnings import warn
from collections import Counter

import numpy as np
from PIL import Image
from skimage.measure import block_reduce
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from helper_functions import *


class Scene(object):
    """Set of satellite images of the same location"""
    def __init__(self, path, dates=None):
        self.path = path
        self.image_dirs = [path + d + "/" for d in os.listdir(path)
                           if ".tar.gz" not in d
                           and d[-1] == "1"
                           and (not dates or
                           get_metadata( path + d + "/", "DATE_ACQUIRED") in dates)
                           ]
        self.length = len(self.image_dirs)

        west_edges = self.query("CORNER_UL_PROJECTION_X_PRODUCT")
        west_edges = np.array([ int(float(x) / 30) for x in west_edges ])
        eastmost_id = np.argmax(west_edges)
        max_edge = west_edges[eastmost_id]
        self.offsets = max_edge - west_edges
        
        aligment_band = generic_band( self.image_dirs[eastmost_id] ).format("1")
        alignment_img = gdal.Open(aligment_band)
        self.geotransform = alignment_img.GetGeoTransform()

    def query(self, substring):
        """Queries metadata of all images within the Scene.

        Uses `get_metadata`, so returns the last Word in the first
        line that contains `substring`, for each file.

        Args:
            substring (str): the substring to search for

        Returns: (list(str))

        """
        return [get_metadata(d, substring) for d in self.image_dirs]

    def __str__(self):
        return "Scene(path={})".format(self.path)

    def __repr__(self):
        return str(self)

    def __len__(self):
        return self.length

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        if self._i < self.length:
            result = (self._i, self.image_dirs[self._i], self.offsets[self._i])
            self._i += 1
            return result
        else:
            raise StopIteration


class Window:
    """ A cropped window into a Scene

    Contains pixel data suitable for time series analysis

    self.location is relative to the Scene, not 'absolute' for the projection.
    """
    def __init__(self, scene, location, shape, pool=(1, 1),
                 channels=(1, 2, 3, 4, 5, 6, 7), pca_components=0.9):
        assert all(shape[i] % pool[i] == 0 for i in [0, 1])
        n_samples = shape[0] * shape[1] // (pool[0] * pool[1])
        n_channels = len(channels) * len(scene)
        self.scene = scene
        self.location = location
        self.shape = shape
        self.pool = pool
        self.data = np.zeros((n_samples, n_channels))
        for i, image_dir, offset in scene:
            band_with_hole = generic_band(image_dir)

            def get_channel(band_id):
                " Gets cropped channel of certain band"
                img = Image.open(band_with_hole.format(band_id))
                arr = np.array(img)
                return self._crop(arr, offset)

            qa_band = get_channel("QA")
            mask = make_mask(qa_band)
            for j in channels:
                channel = get_channel(j)
                channel = self.clean_channel(channel, mask)
                self.data[..., (i * len(channels) + j - 1)] = channel.reshape(-1)
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(self.data)
        self.pca = PCA(n_components=pca_components)
        self.data = self.pca.fit_transform(self.data)

    def clean_channel(self, channel, mask):
        """
        Handles bad data within channel
        """
        channel *= mask
        if self.pool != (1, 1):
            channel = block_reduce(channel, self.pool, np.max)
        # NaN (actually 0) handling:
        # consist of identified obstructed pixels and image padding
        # max pooling may help with poorly identified obstructions
        #
        # note mean() includes the NaNs.
        fill_value = np.true_divide(channel.sum(), (channel!=0).sum())
        channel[channel == 0] = fill_value
        return channel

    def _crop(self, array, x_offset):
        x_lb = self.location[0] + x_offset
        y_lb = self.location[1]
        x_ub = self.shape[0] + x_lb
        y_ub = self.shape[1] + y_lb
        return array[y_lb:y_ub, x_lb:x_ub]

    def align(self, geotif):
        """
        currently assumes UTM Zone 16 and that geotif wholly contains the extent
        
        Raises:
            AssertionError: if GeoTransform data do not sufficiently match
                (may or may not be necessary)

        """
        other_geot = geotif.GetGeoTransform()
        self_geot = self.scene.geotransform
        for i in [1,2,4,5]:
            assert(other_geot[i] == self_geot[i])
        x_lb = (self_geot[0] - other_geot[0]) / other_geot[1] + self.location[0]
        y_lb = (self_geot[3] - other_geot[3]) / (other_geot[5]) + self.location[1]
        img = geotif.ReadAsArray(xoff=x_lb, yoff=y_lb, xsize=self.shape[0], ysize=self.shape[1])
        return img

    def cluster(self, model):
        """
        set self.labels to clusters produced by model
        sorts labels iff n_clusters <= 100
        (probably treats noise in DBSCAN as another cluster)
        """
        labels = model.fit_predict(self.data)
        # Relabel clusters by rank
        if len(set(labels)) > 100:
            warn("Too many clusters: labels are not sorted")
            return
        labels = [string.printable[lbl] for lbl in labels]
        label_counts = Counter(labels)
        ranks = {lbl: rank for rank, (lbl, _) in
                 enumerate(label_counts.most_common())}
        self.labels = np.array([ranks[lbl] for lbl in labels])

    def rebuild(self, labels=None):
        """
        Reconstructs image of original scale from labels array
        """
        if not labels:
            labels = self.labels
        assert len(labels) == len(self.data)
        if self.pool == (1, 1):
            return labels.reshape(self.shape[1], self.shape[0])
        rows = np.array([lbl for lbl in labels for _ in range(self.pool[0])])
        rows = rows.reshape((-1, self.shape[0]))
        result = np.array([row for row in rows for _ in range(self.pool[1])])
        return result

    def __str__(self):
        return "Window(scene={}, location={}, shape={}, pool={}".format(
            self.scene, self.location, self.shape, self.pool)

    def __repr__(self):
        return str(self)
