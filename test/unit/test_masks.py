#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 University of Dundee & Open Microscopy Environment.
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

   Copyright (C) 2019 University of Dundee. All rights reserved.
   Use is subject to license terms supplied in LICENSE.txt
"""
from omero.rtypes import unwrap
import numpy as np
import pytest

from omero_rois import (
    mask_from_binary_image,
    masks_from_label_image,
    NoMaskFound,
)


@pytest.fixture
def binary_image():
    return np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
        ]
    )


@pytest.fixture
def label_image():
    return np.array(
        [
            [0, 0, 0, 2],
            [0, 1, 1, 0],
            [0, 1, 2, 0],
            [0, 0, 0, 0],
        ]
    )


class TestMaskUtils(object):
    @pytest.mark.parametrize(
        "args",
        [{}, {"rgba": (255, 128, 64, 128), "z": 1, "c": 2, "t": 3, "text": "test"}],
    )
    def test_mask_from_binary_image(self, binary_image, args):
        mask = mask_from_binary_image(binary_image, **args)

        assert unwrap(mask.getWidth()) == 2
        assert unwrap(mask.getHeight()) == 2
        assert unwrap(mask.getX()) == 1
        assert unwrap(mask.getY()) == 1

        assert np.array_equal(mask.getBytes(), np.array([224], dtype=np.uint8))

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

    def test_mask_from_binary_full_image(self, binary_image):
        binim = np.ones((4, 4))
        mask = mask_from_binary_image(binim)

        assert unwrap(mask.getWidth()) == 4
        assert unwrap(mask.getHeight()) == 4
        assert np.array_equal(mask.getBytes(), np.array([255, 255], dtype=np.uint8))

    @pytest.mark.parametrize(
        "args",
        [{}, {"rgba": (255, 128, 64, 128), "z": 1, "c": 2, "t": 3, "text": "test"}],
    )
    def test_masks_from_label_image(self, label_image, args):
        masks = masks_from_label_image(label_image, **args)
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

            assert np.array_equal(mask.getBytes(), expected[i][4])

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
    @pytest.mark.parametrize(
        "raise_on_no_mask",
        [
            True,
            False,
        ],
    )
    def test_empty_mask_from_binary_image(self, args, raise_on_no_mask):
        empty_binary_image = np.array([[0]])
        if raise_on_no_mask:
            with pytest.raises(NoMaskFound):
                mask = mask_from_binary_image(
                    empty_binary_image, raise_on_no_mask=raise_on_no_mask, **args
                )
        else:
            mask = mask_from_binary_image(
                empty_binary_image, raise_on_no_mask=raise_on_no_mask, **args
            )
            assert unwrap(mask.getWidth()) == 0
            assert unwrap(mask.getHeight()) == 0
            assert unwrap(mask.getX()) == 0
            assert unwrap(mask.getY()) == 0
            assert np.array_equal(mask.getBytes(), [])

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
