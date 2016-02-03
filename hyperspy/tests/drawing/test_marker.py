# Copyright 2007-2016 The HyperSpy developers
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


import numpy as np
import nose.tools

from hyperspy.signals import Image
from hyperspy.utils import markers


class Test_markers:

    def test_get_data(self):
        s = Image(np.zeros([3, 2, 2]))
        m = markers.line_segment(x1=range(3), x2=range(3), y1=1.3, y2=1.5)
        m.axes_manager = s.axes_manager
        nose.tools.assert_true(m.get_data_position('x1') == 0)
        nose.tools.assert_true(m.get_data_position('y1') == 1.3)
        s.axes_manager[0].index = 2
        nose.tools.assert_true(m.get_data_position('x1') == 2)
        nose.tools.assert_true(m.get_data_position('y1') == 1.3)

    def test_get_data_array(self):
        s = Image(np.zeros([2, 2, 2, 2]))
        m = markers.line_segment(x1=[[1.1, 1.2], [1.3, 1.4]], x2=1.1, y1=1.3,
                                 y2=1.5)
        m.axes_manager = s.axes_manager
        nose.tools.assert_true(m.get_data_position('x1') == 1.1)
        s.axes_manager[0].index = 1
        nose.tools.assert_true(m.get_data_position('x1') == 1.2)
        s.axes_manager[1].index = 1
        nose.tools.assert_true(m.get_data_position('x1') == 1.4)

    def test_set_get_data(self):
        m = markers.point(x=0, y=1.3)
        nose.tools.assert_true(m.data['x1'] == 0)
        nose.tools.assert_true(m.data['y1'] == 1.3)
        m.add_data(y1=0.3)
        nose.tools.assert_true(m.data['x1'] == 0)
        nose.tools.assert_true(m.data['y1'] == 0.3)
        m.set_data(y1=1.3)
        nose.tools.assert_true(m.data['x1'][()][()] is None)
        nose.tools.assert_true(m.data['y1'] == 1.3)
        nose.tools.assert_true(m.data['x1'].dtype == np.dtype('O'))
        m.add_data(y1=[1, 2])
        nose.tools.assert_true(m.data['y1'][()].shape == (2,))

    def test_markers_properties(self):
        m = markers.text(x=1, y=2, text='a')
        m.set_marker_properties(fontsize=30, color='red')
        nose.tools.assert_true(m.marker_properties ==
                               {'color': 'red', 'fontsize': 30})

    def test_auto_update(self):
        m = markers.text(y=1, x=2, text='a')
        nose.tools.assert_true(m.auto_update is False)
        m = markers.text(y=[1, 2], x=2, text='a')
        nose.tools.assert_true(m.auto_update is True)
        m.add_data(y1=1)
        nose.tools.assert_true(m.auto_update is False)
        m.add_data(y1=[1, 2])
        nose.tools.assert_true(m.auto_update is True)
