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

import numpy as np
from omero.gateway import ColorHolder
from omero.model import MaskI, ShapeI
from typing import Tuple, List, Dict, Set
from omero.rtypes import (
    rdouble,
    rint,
    rstring,
    unwrap
)
import re

# Mapping of dimension names to axes in the image array
DIMENSION_ORDER: Dict[str, int] = {
    "T": 0,
    "C": 1,
    "Z": 2,
    "Y": 3,
    "X": 4,
}

OME_MODEL_POINT_LIST_RE = re.compile(r"([\d.]+),([\d.]+)")


class NoMaskFound(ValueError):
    """
    Exception thrown when no foreground pixels were found in a mask
    """

    def __init__(self, msg="No mask found"):
        super(Exception, self).__init__(msg)


class InvalidBinaryImage(ValueError):
    """
    Exception thrown when invalid labels are found
    """

    def __init__(self, msg="Invalid labels found"):
        super(Exception, self).__init__(msg)


def mask_from_binary_image(
    binim, rgba=None, z=None, c=None, t=None, text=None, raise_on_no_mask=True
):
    """
    Create a mask shape from a binary image (background=0)

    :param numpy.array binim: Binary 2D array, must contain values [0, 1] only
    :param rgba int-4-tuple: Optional (red, green, blue, alpha) colour
    :param z: Optional Z-index for the mask
    :param c: Optional C-index for the mask
    :param t: Optional T-index for the mask
    :param text: Optional text for the mask
    :param raise_on_no_mask: If True (default) throw an exception if no mask
           found, otherwise return an empty Mask
    :return: An OMERO mask
    :raises NoMaskFound: If no labels were found
    :raises InvalidBinaryImage: If the maximum labels is greater than 1
    """

    # Find bounding box to minimise size of mask
    xmask = binim.sum(0).nonzero()[0]
    ymask = binim.sum(1).nonzero()[0]
    if any(xmask) and any(ymask):
        x0 = min(xmask)
        w = max(xmask) - x0 + 1
        y0 = min(ymask)
        h = max(ymask) - y0 + 1
        submask = binim[y0 : (y0 + h), x0 : (x0 + w)]
        if not np.array_equal(np.unique(submask), [0, 1]) and not np.array_equal(
            np.unique(submask), [1]
        ):
            raise InvalidBinaryImage()
    else:
        if raise_on_no_mask:
            raise NoMaskFound()
        x0 = 0
        w = 0
        y0 = 0
        h = 0
        submask = []

    mask = MaskI()
    # BUG in older versions of Numpy:
    # https://github.com/numpy/numpy/issues/5377
    # Need to convert to an int array
    # mask.setBytes(np.packbits(submask))
    mask.setBytes(np.packbits(np.asarray(submask, dtype=int)))
    mask.setWidth(rdouble(w))
    mask.setHeight(rdouble(h))
    mask.setX(rdouble(x0))
    mask.setY(rdouble(y0))

    if rgba is not None:
        ch = ColorHolder.fromRGBA(*rgba)
        mask.setFillColor(rint(ch.getInt()))
    if z is not None:
        mask.setTheZ(rint(z))
    if c is not None:
        mask.setTheC(rint(c))
    if t is not None:
        mask.setTheT(rint(t))
    if text is not None:
        mask.setTextValue(rstring(text))

    return mask


def masks_from_label_image(
    labelim, rgba=None, z=None, c=None, t=None, text=None, raise_on_no_mask=True
):
    """
    Create mask shapes from a label image (background=0)

    :param numpy.array labelim: 2D label array
    :param rgba int-4-tuple: Optional (red, green, blue, alpha) colour
    :param z: Optional Z-index for the mask
    :param c: Optional C-index for the mask
    :param t: Optional T-index for the mask
    :param text: Optional text for the mask
    :param raise_on_no_mask: If True (default) throw an exception if no mask
           found, otherwise return an empty Mask
    :return: A list of OMERO masks in label order ([] if no labels found)

    """
    masks = []
    for i in range(1, labelim.max() + 1):
        mask = mask_from_binary_image(
            labelim == i, rgba, z, c, t, text, raise_on_no_mask
        )
        masks.append(mask)
    return masks


def shape_to_binary_image(
    self, shape: ShapeI
) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """
    Convert an OMERO shape to a binary image

    :param shape ShapeI: An OMERO shape
    :return: tuple of
            - Binary mask
            - (T, C, Z, Y, X, w, h) tuple of mask settings (T, C, Z may be
            None)
    """
    if isinstance(shape, MaskI):
        return _mask_to_binary_image(shape)
    return _polygon_to_binary_image(shape)


def _mask_to_binary_image(
    mask: ShapeI, dtype=np.bool
) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """
    Convert an OMERO mask to a binary image

    :param mask MaskI: An OMERO mask
    :param dtype: Data type for the binary image
    :return: tuple of
            - Binary mask
            - (T, C, Z, Y, X, w, h) tuple of mask settings (T, C, Z may be
            None)
    """

    t = unwrap(mask.theT)
    c = unwrap(mask.theC)
    z = unwrap(mask.theZ)

    x = int(mask.x.val)
    y = int(mask.y.val)
    w = int(mask.width.val)
    h = int(mask.height.val)

    mask_packed = mask.getBytes()
    # convert bytearray into something we can use
    intarray = np.fromstring(mask_packed, dtype=np.uint8)
    binarray = np.unpackbits(intarray).astype(dtype)
    # truncate and reshape
    binarray = np.reshape(binarray[: (w * h)], (h, w))

    return binarray, (t, c, z, y, x, h, w)


def _polygon_to_binary_image(
    self, polygon: ShapeI
) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """
    Convert an OMERO polygon to a binary image

    :param polygon ShapeI: An OMERO polygon
    :return: tuple of
            - Binary mask
            - (T, C, Z, Y, X, w, h) tuple of mask settings (T, C, Z may be
            None)
    """
    from skimage.draw import polygon

    t = unwrap(polygon.theT)
    c = unwrap(polygon.theC)
    z = unwrap(polygon.theZ)

    # "10,20, 50,150, 200,200, 250,75"
    points = unwrap(polygon.points).strip()
    coords = OME_MODEL_POINT_LIST_RE.findall(points)
    x_coords = np.array([int(round(float(xy[0]))) for xy in coords])
    y_coords = np.array([int(round(float(xy[1]))) for xy in coords])

    # bounding box of polygon
    x = x_coords.min()
    y = y_coords.min()
    w = x_coords.max() - x
    h = y_coords.max() - y

    img = np.zeros((h, w), dtype=self.dtype)

    # coords *within* bounding box
    x_coords = x_coords - x
    y_coords = y_coords - y

    pixels = polygon(y_coords, x_coords, img.shape)
    img[pixels] = 1

    return img, (t, c, z, y, x, h, w)


def masks_to_labels(
    self,
    masks: List[MaskI],
    mask_shape: Tuple[int, ...],
    ignored_dimensions: Set[str] = None,
    check_overlaps: bool = True,
) -> Tuple[np.ndarray, Dict[int, str], Dict[int, Dict]]:
    """
    :param masks [MaskI]: Iterable container of OMERO masks
    :param mask_shape 5-tuple: the image dimensions (T, C, Z, Y, X), taking
        into account `ignored_dimensions`

    :param ignored_dimensions set(char): Ignore these dimensions and set
        size to 1

    :param check_overlaps bool: Whether to check for overlapping masks or
        not

    :return: Label image with size `mask_shape` as well as color metadata
        and dict of other properties.
    """

    # FIXME: hard-coded dimensions
    assert len(mask_shape) > 3
    size_t: int = mask_shape[0]
    size_c: int = mask_shape[1]
    size_z: int = mask_shape[2]
    ignored_dimensions = ignored_dimensions or set()

    labels = np.zeros(mask_shape, np.int64)

    for d in "TCZYX":
        if d in ignored_dimensions:
            assert (
                labels.shape[DIMENSION_ORDER[d]] == 1
            ), f"Ignored dimension {d} should be size 1"
        assert (
            labels.shape == mask_shape
        ), f"Invalid label shape: {labels.shape}, expected {mask_shape}"

    fillColors: Dict[int, str] = {}
    properties: Dict[int, Dict] = {}

    for count, shapes in enumerate(masks):
        for shape in shapes:
            # Using ROI ID allows stitching label from multiple images
            # into a Plate and not creating duplicates from different iamges.
            # All shapes will be the same value (color) for each ROI
            shape_value = shape.roi.id.val
            properties[shape_value] = {
                "omero:shapeId": shape.id.val,
                "omero:roiId": shape.roi.id.val,
            }
            if shape.textValue:
                properties[shape_value]["omero:text"] = unwrap(shape.textValue)
            if shape.fillColor:
                fillColors[shape_value] = unwrap(shape.fillColor)
            binim_yx, (t, c, z, y, x, h, w) = shape_to_binary_image(shape)
            for i_t in self._get_indices(ignored_dimensions, "T", t, size_t):
                for i_c in self._get_indices(ignored_dimensions, "C", c, size_c):
                    for i_z in self._get_indices(
                        ignored_dimensions, "Z", z, size_z
                    ):
                        if check_overlaps and np.any(
                            np.logical_and(
                                labels[
                                    i_t, i_c, i_z, y : (y + h), x : (x + w)
                                ].astype(np.bool),
                                binim_yx,
                            )
                        ):
                            raise Exception(
                                f"Mask {shape_value} overlaps with existing labels"
                            )
                        # ADD to the array, so zeros in our binarray don't
                        # wipe out previous shapes
                        labels[i_t, i_c, i_z, y : (y + h), x : (x + w)] += (
                            binim_yx * shape_value
                        )

    return labels, fillColors, properties