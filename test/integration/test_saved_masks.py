#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2020 University of Dundee & Open Microscopy Environment.
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

"""
   Test of ROI mask utils

   Copyright (C) 2020 University of Dundee. All rights reserved.
   Use is subject to license terms supplied in LICENSE.txt
"""

import omero.clients
from omero.gateway import BlitzGateway
from omero.model import RoiI
from omero.testlib import ITest
from omero.rtypes import unwrap
import numpy as np
import pytest

from omero_rois import (
    mask_from_binary_image,
    masks_from_label_image,
    masks_to_labels,
    shape_to_binary_image,
)


@pytest.fixture
def binary_image():
    return np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )


@pytest.fixture
def label_image():
    return np.array(
        [
            [0, 0, 0, 2],
            [0, 1, 1, 0],
            [0, 1, 2, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )


class TestSavedMasks(ITest):
    def setup_method(self):
        self.conn = BlitzGateway(client_obj=self.client)

    def save_and_return_masks(self, im, masks):
        roi = RoiI()
        for mask in masks:
            roi.addShape(mask)

        us = self.conn.getUpdateService()
        roi.setImage(im._obj)
        roi = us.saveAndReturnObject(roi)
        assert roi

        rois = self.conn.getRoiService().findByImage(im.id, None)
        assert len(rois.rois) == 1
        shapes = rois.rois[0].copyShapes()
        assert len(shapes) == len(masks)
        assert all((type(mask) == omero.model.MaskI) for mask in shapes)

        return shapes

    @pytest.mark.parametrize(
        "args",
        [{}, {"rgba": (255, 128, 64, 128), "z": 1, "c": 2, "t": 3, "text": "test"}],
    )
    def test_mask_from_binary_image(self, binary_image, args):
        im = self.conn.createImageFromNumpySeq(
            iter([binary_image]), "omero-rois-integration"
        )
        # Reload
        im = self.conn.getObject("Image", im.id)
        px = im.getPrimaryPixels().getPlane()

        mask = mask_from_binary_image(px, **args)
        mask = self.save_and_return_masks(im, [mask])[0]

        # The rest of this is more or less the same as the unit test

        assert unwrap(mask.getWidth()) == 2
        assert unwrap(mask.getHeight()) == 2
        assert unwrap(mask.getX()) == 1
        assert unwrap(mask.getY()) == 1

        assert np.array_equal(
            np.frombuffer(mask.getBytes(), np.uint8), np.array([224], dtype=np.uint8)
        )

        if args:
            assert unwrap(mask.getTheZ()) == 1
            assert unwrap(mask.getTheC()) == 2
            assert unwrap(mask.getTheT()) == 3
            assert unwrap(mask.getTextValue()) == "test"
        else:
            assert unwrap(mask.getTheZ()) is None
            assert unwrap(mask.getTheC()) is None
            assert unwrap(mask.getTheT()) is None
            assert unwrap(mask.getTextValue()) is None

    def test_mask_from_binary_full_image(self):
        binim = np.ones((4, 4), dtype=np.uint8)
        im = self.conn.createImageFromNumpySeq(iter([binim]), "omero-rois-integration")
        # Reload
        im = self.conn.getObject("Image", im.id)
        px = im.getPrimaryPixels().getPlane()

        mask = mask_from_binary_image(px)
        mask = self.save_and_return_masks(im, [mask])[0]

        # The rest of this is more or less the same as the unit test

        assert unwrap(mask.getWidth()) == 4
        assert unwrap(mask.getHeight()) == 4
        assert np.array_equal(
            np.frombuffer(mask.getBytes(), np.uint8),
            np.array([255, 255], dtype=np.uint8),
        )

    @pytest.mark.parametrize(
        "args",
        [{}, {"rgba": (255, 128, 64, 128), "z": 1, "c": 2, "t": 3, "text": "test"}],
    )
    def test_masks_from_label_image(self, label_image, args):
        im = self.conn.createImageFromNumpySeq(
            iter([label_image]), "omero-rois-integration"
        )
        # Reload
        im = self.conn.getObject("Image", im.id)
        px = im.getPrimaryPixels().getPlane()

        masks = masks_from_label_image(px, **args)
        masks = self.save_and_return_masks(im, masks)

        # The rest of this is more or less the same as the unit test

        expected = (
            # w, h, x, y, bytes
            (2, 2, 1, 1, np.array([224], dtype=np.uint8)),
            (2, 3, 2, 0, np.array([72], dtype=np.uint8)),
        )

        assert len(masks) == 2

        for i, mask in enumerate(masks):
            assert unwrap(mask.getWidth()) == expected[i][0]
            assert unwrap(mask.getHeight()) == expected[i][1]
            assert unwrap(mask.getX()) == expected[i][2]
            assert unwrap(mask.getY()) == expected[i][3]

            assert np.array_equal(
                np.frombuffer(mask.getBytes(), np.uint8), expected[i][4]
            )

            if args:
                assert unwrap(mask.getTheZ()) == 1
                assert unwrap(mask.getTheC()) == 2
                assert unwrap(mask.getTheT()) == 3
                assert unwrap(mask.getTextValue()) == "test"
            else:
                assert unwrap(mask.getTheZ()) is None
                assert unwrap(mask.getTheC()) is None
                assert unwrap(mask.getTheT()) is None
                assert unwrap(mask.getTextValue()) is None

    @pytest.mark.parametrize(
        "args",
        [{}, {"rgba": (255, 128, 64, 128), "z": 1, "c": 2, "t": 3, "text": "test"}],
    )
    def test_empty_mask_from_binary_image(self, args):
        empty_binary_image = np.array([[0]], dtype=np.uint8)
        raise_on_no_mask = False
        im = self.conn.createImageFromNumpySeq(
            iter([empty_binary_image]), "omero-rois-integration"
        )
        # Reload
        im = self.conn.getObject("Image", im.id)
        px = im.getPrimaryPixels().getPlane()

        mask = mask_from_binary_image(px, raise_on_no_mask=raise_on_no_mask, **args)
        mask = self.save_and_return_masks(im, [mask])[0]

        # The rest of this is more or less the same as the unit test

        assert unwrap(mask.getWidth()) == 0
        assert unwrap(mask.getHeight()) == 0
        assert unwrap(mask.getX()) == 0
        assert unwrap(mask.getY()) == 0
        assert np.array_equal(np.frombuffer(mask.getBytes(), np.uint8), [])

        if args:
            assert unwrap(mask.getTheZ()) == 1
            assert unwrap(mask.getTheC()) == 2
            assert unwrap(mask.getTheT()) == 3
            assert unwrap(mask.getTextValue()) == "test"
        else:
            assert unwrap(mask.getTheZ()) is None
            assert unwrap(mask.getTheC()) is None
            assert unwrap(mask.getTheT()) is None
            assert unwrap(mask.getTextValue()) is None

    def test_shape_to_binary_image(self, label_image):
        im = self.conn.createImageFromNumpySeq(
            iter([label_image]), "test_shape_to_binary_image"
        )
        im = self.conn.getObject("Image", im.id)
        px = im.getPrimaryPixels().getPlane()

        masks = masks_from_label_image(px)
        self.save_and_return_masks(im, masks)[0]

        # load masks and convert back to labels
        rois = self.conn.getRoiService().findByImage(im.id, None)
        assert len(rois.rois) == 1
        shapes = rois.rois[0].copyShapes()
        assert len(shapes) == 2
        assert type(shapes[0]) == omero.model.MaskI

        expected_arrs = [
            np.array(
                [
                    [1, 1],
                    [1, 0],
                ],
                dtype=np.uint8,
            ),
            np.array(
                [
                    [0, 1],
                    [0, 0],
                    [1, 0],
                ],
                dtype=np.uint8,
            ),
        ]
        for expected_arr, omero_shape in zip(expected_arrs, shapes):
            shape_info = shape_to_binary_image(omero_shape)
            np_bool_data = shape_info[0]
            t, c, z, y, x, h, w = shape_info[1]
            assert (h, w) == expected_arr.shape
            assert (y, x) == (omero_shape.getY().val, omero_shape.getX().val)
            assert np.array_equal(np_bool_data * 1, expected_arr)

    def test_masks_to_labels(self) -> None:
        """Test export of a Zarr image."""
        size_xy = 20
        size_z = 5
        size_c = 2
        img_shape = (1, size_c, size_z, size_xy, size_xy)  # (T, C, Z, Y, X)
        blank_image = np.zeros((size_xy, size_xy), dtype=np.uint8)
        im = self.conn.createImageFromNumpySeq(
            iter([blank_image] * (size_z * size_c)),
            "test_masks_to_labels",
            sizeZ=size_z,
            sizeC=size_c,
        )

        # Create a mask
        from skimage.data import binary_blobs

        blobs = binary_blobs(length=size_xy, volume_fraction=0.1, n_dim=2).astype(
            "int8"
        )
        red = [255, 0, 0, 255]

        # create ROIs in OMERO... - one per channel, with shape on each Z plane
        new_rois = []
        for theC in range(size_c):
            roi = RoiI()
            roi.setImage(omero.model.ImageI(im.id, False))
            for theZ in range(size_z):
                mask = mask_from_binary_image(blobs, rgba=red, z=theZ, c=theC, t=0)
                roi.addShape(mask)
            updateService = self.client.sf.getUpdateService()
            new_rois.append(updateService.saveAndReturnObject(roi))

        rois = self.conn.getRoiService().findByImage(im.id, None)
        assert len(rois.rois) == size_c
        shapes = []
        for r in rois.rois:
            shapes.extend(r.copyShapes())

        labels, fillColors, properties = masks_to_labels(shapes, img_shape)
        print("labels, fillColors, properties", labels, fillColors, properties)
        assert labels.shape == img_shape
        assert labels.dtype == np.int64
        # The value of each label array corresponds to ROI ID
        assert labels.max() == len(new_rois)

        for theC in range(size_c):
            for theZ in range(size_z):
                mask = blobs.astype(bool)
                expected_labels = np.where(mask, theC + 1, 0)
                assert np.array_equal(labels[0, theC, theZ], expected_labels)
