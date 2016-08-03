# Copyright 2007-2016 The HyperSpy developers
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


import numpy as np
from nose.tools import (assert_true,
                        assert_false,
                        assert_equal,
                        raises)
import nose.tools as nt
from hyperspy.component import Parameter
from hyperspy.exceptions import NavigationDimensionError
from unittest import mock


class Dummy:

    def __init__(self):
        self.value = 1

    def add_one(self):
        self.value += 1


class DummyAxesManager:
    navigation_shape = ()
    indices = ()

    @property
    def _navigation_shape_in_array(self):
        return self.navigation_shape[::-1]


class TestParameterLen1:

    def setUp(self):
        self.par = Parameter()

    def test_set_value(self):
        self.par.value = 2
        assert_equal(self.par.value, 2)

    @raises(ValueError)
    def test_set_value_wrong_length(self):
        self.par.value = (2, 2)

    def test_set_value_bounded(self):
        self.par.bmin = 1
        self.par.bmax = 3
        self.par.ext_bounded = True
        assert_equal(self.par.value, 1)
        self.par.value = 1.5
        assert_equal(self.par.value, 1.5)
        self.par.value = 0.5
        assert_equal(self.par.value, 1)
        self.par.value = 4
        assert_equal(self.par.value, 3)
        self.par.bmax = 2
        assert_equal(self.par.value, 2)
        self.par.value = 1.5
        self.par.bmin = 1.6
        assert_equal(self.par.value, 1.6)

    def test_ext_force_positive(self):
        self.par.ext_bounded = True
        self.par.ext_force_positive = True
        self.par.value = -1.5
        assert_equal(self.par.value, 1.5)
        self.par.bmax = 2
        self.par.value = -3
        assert_equal(self.par.value, 2)

    def test_number_of_elements(self):
        assert_equal(len(self.par), 1)

    def test_default_value(self):
        assert_equal(self.par.value, 0)

    def test_connect_disconnect(self):
        dummy = Dummy()
        self.par.events.value_changed.connect(dummy.add_one, [])
        self.par.value = 1
        assert_equal(dummy.value, 2)

        # Setting the same value should not call the connected functions
        self.par.value = 1
        assert_equal(dummy.value, 2)

        # After disconnecting dummy.value should not change
        self.par.events.value_changed.disconnect(dummy.add_one)
        self.par.value = 2
        assert_equal(dummy.value, 2)

    def test_map_size0(self):
        self.par._axes_manager = DummyAxesManager()
        self.par._create_array()
        self.par.value = 1
        self.par.std = 0.1
        self.par.store_current_value_in_array()
        assert_equal(self.par.map['values'][0], 1)
        assert_true(self.par.map['is_set'][0])
        assert_equal(self.par.map['std'][0], 0.1)

    def test_map_size1(self):
        self.par._axes_manager = DummyAxesManager()
        self.par._axes_manager.navigation_shape = [1, ]
        self.par._create_array()
        self.par.value = 1
        self.par.std = 0.1
        self.par.store_current_value_in_array()
        assert_equal(self.par.map['values'][0], 1)
        assert_true(self.par.map['is_set'][0])
        assert_equal(self.par.map['std'][0], 0.1)

    def test_map_size2(self):
        self.par._axes_manager = DummyAxesManager()
        self.par._axes_manager.navigation_shape = [2, ]
        self.par._axes_manager.indices = (1,)
        self.par._create_array()
        self.par.value = 1
        self.par.std = 0.1
        self.par.store_current_value_in_array()
        assert_equal(self.par.map['values'][1], 1)
        assert_true(self.par.map['is_set'][1])
        assert_equal(self.par.map['std'][1], 0.1)


class TestParameterLen2:

    def setUp(self):
        self.par = Parameter()
        self.par._number_of_elements = 2

    def test_set_value(self):
        self.par.value = (2, 2)
        assert_equal(self.par.value, (2, 2))

    @raises(ValueError)
    def test_set_value_wrong_length(self):
        self.par.value = 2

    @raises(ValueError)
    def test_set_value_wrong_length2(self):
        self.par.value = (2, 2, 2)

    def test_set_value_bounded(self):
        self.par.bmin = 1
        self.par.bmax = 3
        self.par.ext_bounded = True
        self.par.value = (1.5, 1.5)
        assert_equal(self.par.value, (1.5, 1.5))
        self.par.value = (0.5, 0.5)
        assert_equal(self.par.value, (1, 1))
        self.par.value = (4, 4)
        assert_equal(self.par.value, (3, 3))

    def test_ext_force_positive(self):
        self.par.ext_bounded = True
        self.par.ext_force_positive = True
        self.par.value = (2, -1.5)
        assert_equal(self.par.value, (2, 1.5))

    def test_number_of_elements(self):
        assert_equal(len(self.par), 2)

    def test_default_value(self):
        assert_equal(self.par.value, (0, 0))

    def test_connect_disconnect(self):
        dummy = Dummy()
        self.par.events.value_changed.connect(dummy.add_one, [])
        self.par.value = (1, 1)
        assert_equal(dummy.value, 2)

        # Setting the same value should not call the connected functions
        self.par.value = (1, 1)
        assert_equal(dummy.value, 2)

        # After disconnecting dummy.value should not change
        self.par.events.value_changed.disconnect(dummy.add_one)
        self.par.value = (2, 2)
        assert_equal(dummy.value, 2)

    def test_map_size0(self):
        self.par._axes_manager = DummyAxesManager()
        self.par._create_array()
        self.par.value = (1, 1)
        self.par.std = (0.1, 0.1)
        self.par.store_current_value_in_array()
        assert_equal(tuple(self.par.map['values'][0]), (1, 1))
        assert_true(self.par.map['is_set'][0])
        assert_equal(tuple(self.par.map['std'][0]), (0.1, 0.1))

    def test_map_size1(self):
        self.par._axes_manager = DummyAxesManager()
        self.par._axes_manager.navigation_shape = [1, ]
        self.par._create_array()
        self.par.value = (1, 1)
        self.par.std = (0.1, 0.1)
        self.par.store_current_value_in_array()
        assert_equal(tuple(self.par.map['values'][0]), (1, 1))
        assert_true(self.par.map['is_set'][0])
        assert_equal(tuple(self.par.map['std'][0]), (0.1, 0.1))

    def test_map_size2(self):
        self.par._axes_manager = DummyAxesManager()
        self.par._axes_manager.navigation_shape = [2, ]
        self.par._axes_manager.indices = (1,)
        self.par._create_array()
        self.par.value = (1, 1)
        self.par.std = (0.1, 0.1)
        self.par.store_current_value_in_array()
        assert_equal(tuple(self.par.map['values'][1]), (1, 1))
        assert_true(self.par.map['is_set'][1])
        assert_equal(tuple(self.par.map['std'][1]), (0.1, 0.1))

    def test_is_tuple(self):
        self.par.value = np.array((1, 2))
        assert_true(isinstance(self.par.value, tuple))
        self.par.value = [1, 2]
        assert_true(isinstance(self.par.value, tuple))


class TestParameterTwin:

    def setUp(self):
        self.p1 = Parameter()
        self.p2 = Parameter()

    def test_slave_fixed(self):
        assert_true(self.p2.free)
        self.p2.twin = self.p1
        assert_false(self.p2.free)

    def test_twin_value(self):
        self.p2.twin = self.p1
        self.p1.value = 3
        assert_equal(self.p1.value, self.p2.value)
        self.p2.twin = None
        assert_equal(self.p1.value, self.p2.value)
        self.p1.value = 2
        assert_equal(3, self.p2.value)

    def test_twin_value_bounded(self):
        self.p2.bmax = 2
        self.p2.ext_bounded = True
        self.p2.twin = self.p1
        self.p1.value = 3
        assert_equal(self.p1.value, self.p2.value)
        self.p2.twin = None
        assert_equal(self.p2.value, self.p2.bmax)

    def test_twin_function(self):
        self.p2.twin_function = lambda x: x + 2
        self.p2.twin_inverse_function = lambda x: x - 2
        self.p2.twin = self.p1
        assert_equal(self.p1.value, self.p2.value - 2)
        self.p2.value = 10
        assert_equal(self.p1.value, 8)

    def test_inherit_connections(self):
        dummy = Dummy()
        self.p2.events.value_changed.connect(dummy.add_one, [])
        self.p2.twin = self.p1
        self.p1.value = 2
        assert_equal(dummy.value, 2)
        # The next line calls add_one -> value = 3
        self.p2.twin = None
        # Next one shouldn't call add_one -> value = 3
        self.p1.value = 4
        assert_equal(dummy.value, 3)
        self.p2.value = 10
        assert_equal(dummy.value, 4)


class TestGeneralMethods:

    def setUp(self):
        self.par = Parameter()
        self.par._axes_manager = mock.MagicMock()
        self.par.map = np.array(
            [(a, b, c) for a, b, c in zip([1, 3, 5], [2, 4, 6], [0, 0, 0])],
            dtype=[('values', 'float'), ('std', 'float'), ('is_set', bool)])

    def test_as_signal(self):
        par = self.par

        # additional setup
        par._axes_manager._get_navigation_axes_dicts.return_value = [
            {'name': 'one', 'navigate': True,
             'offset': 0.0, 'scale':
             1.0, 'size': 3,
             'units': 'bar'}, ]
        par._number_of_elements = 2
        par.component = mock.MagicMock()
        par.component.active_is_multidimensional = True
        par.component._active_array = np.array([1, 0, 1], dtype=bool)

        # testing
        s = par.as_signal('std')
        np.testing.assert_array_equal(s.data, np.array([2, np.nan, 6]))
        nt.assert_equal(s.axes_manager[-1].name, 'one')
        nt.assert_true(par._axes_manager._get_navigation_axes_dicts.called)
        nt.assert_equal(len(s.axes_manager._axes), 2)
        nt.assert_false(s.axes_manager[-1].navigate)
        nt.assert_true(s.axes_manager[0].navigate)
        nt.assert_equal(s.axes_manager[0].size, 2)

    def test_store_current_values_normal_indices(self):
        par = self.par
        par._axes_manager.indices = (1,)
        par.value = 3.5
        par.std = 4.5
        par.store_current_value_in_array()
        nt.assert_true(par.map['is_set'][1])
        nt.assert_equal(par.map['std'][1], 4.5)
        nt.assert_equal(par.map['values'][1], 3.5)

    def test_store_current_values_no_indices(self):
        par = self.par
        par._axes_manager.indices = ()
        par.value = 3.5
        par.std = 4.5
        par.store_current_value_in_array()
        nt.assert_true(par.map['is_set'][0])
        nt.assert_equal(par.map['std'][0], 4.5)
        nt.assert_equal(par.map['values'][0], 3.5)
