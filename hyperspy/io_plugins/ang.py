# -*- coding: utf-8 -*-
# Copyright 2007-2015 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.

# The details of the format were taken from
# http://www.biochem.mpg.de/doc_tom/TOM_Release_2008/IOfun/tom_mrcread.html
# and http://ami.scripps.edu/software/mrctools/mrc_specification.php


import numpy as np
from traits.api import Undefined


# Plugin characteristics
# ----------------------
format_name = 'ANG'
description = ''
full_support = False
# Recognised file extension
file_extensions = ['ang']
default_extension = 0

# Writing capabilities
writes = False


def file_reader(filename, *args, **kwds):
    # Format of data row:
    # R0 R1 R2 X Y IQ CI Phase Index
    rawdata = np.loadtxt(filename, unpack=True)
    dx = rawdata[3, 1] - rawdata[3, 0]
    unique_Y = np.unique(rawdata[4, :])
    dy = unique_Y[1] - unique_Y[0]

    data = np.concatenate((rawdata[0:3, :], rawdata[5:, :]))
    nX = len(np.unique(rawdata[3, :]))
    nY = len(unique_Y)
    data = data.reshape((data.shape[0], nY, nX))

    units = [Undefined, 'nm', 'nm']
    names = ['data', 'x', 'y']
    offsets = np.concatenate(([1], rawdata[3:5, 0]))
    scales = [1, dx, dy]
    navigate = [True, False, False]

    axes = [
        {
            'size': data.shape[i],
            'index_in_array': i,
            'name': names[i],
            'scale': scales[i],
            'offset': offsets[i],
            'units': units[i],
            'navigate': navigate[i], }
        for i in xrange(3)]

    dictionary = {'data': data,
                  'axes': axes,
                  'metadata': {},
                  'original_metadata': {}, }

    return [dictionary, ]
