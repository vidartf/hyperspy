# Copyright 2007-2012 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <http://www.gnu.org/licenses/>.


import nose.tools as nt
import numpy as np
from hyperspy.components import Gaussian
from hyperspy.signal import Signal


class TestParametersAsSignals:

    def setUp(self):
        self.gaussian = Gaussian()
        self.gaussian._axes_manager = Signal(np.empty((3, 3, 1))).axes_manager

    def test_always_active(self):
        g = self.gaussian
        g.active_is_multidimensional = False
        g._create_arrays()
        nt.assert_true(
            np.all(
                g.A.as_signal('values').data == np.zeros(
                    (3, 3))))

    def test_some_inactive(self):
        g = self.gaussian
        g.active_is_multidimensional = True
        g._create_arrays()
        g._active_array[2, 0] = False
        g._active_array[0, 0] = False
        nt.assert_true(
            np.isnan(g.A.as_signal('values').data[[0, 2], [0]]).all())
