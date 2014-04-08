# -*- coding: utf-8 -*-
# Copyright 2007-2011 The HyperSpy developers
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

# custom exceptions
from hyperspy import messages
from hyperspy.exceptions import NoInteractiveError
from hyperspy.defaults_parser import preferences
from hyperspy.gui.tools import SpectrumRangeSelector


def simple_decorator(decorator):
    """This decorator can be used to turn simple functions
    into well-behaved decorators, so long as the decorators
    are fairly simple. If a decorator expects a function and
    returns a function (no descriptors), and if it doesn't
    modify function attributes or docstring, then it is
    eligible to use this. Simply apply @simple_decorator to
    your decorator and it will automatically preserve the
    docstring and function attributes of functions to which
    it is applied.

    This decorator was taken from:
    http://wiki.python.org/moin/PythonDecoratorLibrary"""
    def new_decorator(f):
        g = decorator(f)
        g.__name__ = f.__name__
        g.__doc__ = f.__doc__
        g.__dict__.update(f.__dict__)
        return g
    # Now a few lines needed to make simple_decorator itself
    # be a well-behaved decorator.
    new_decorator.__name__ = decorator.__name__
    new_decorator.__doc__ = decorator.__doc__
    new_decorator.__dict__.update(decorator.__dict__)
    return new_decorator


@simple_decorator
def only_interactive(cm):
    def wrapper(*args, **kwargs):
        if preferences.General.interactive is True:
            return cm(*args, **kwargs)
        else:
            raise NoInteractiveError
    return wrapper


@simple_decorator
def interactive_range_selector(cm):
    def wrapper(self, *args, **kwargs):
        if preferences.General.interactive is True and not args and not kwargs:
            range_selector = SpectrumRangeSelector(self)
            range_selector.on_close.append((cm, self))
            if preferences.General.modal is True:
                range_selector.edit_traits(kind='livemodal')
            else:
                range_selector.edit_traits()
        else:
            cm(self, *args, **kwargs)
    return wrapper


@simple_decorator
def auto_replot(cm):
    def wrapper(self, *args, **kwargs):
        if self.auto_replot is True:
            to_return = cm(self, *args, **kwargs)
            self._replot()
            return to_return
        else:
            return cm(self, *args, **kwargs)
    return wrapper


@simple_decorator
def do_not_replot(cm):
    def wrapper(self, *args, **kwargs):
        if self.auto_replot is True:
            self.auto_replot = False
            to_return = cm(self, *args, **kwargs)
            self.auto_replot = True
            return to_return
        else:
            return cm(self, *args, **kwargs)
    return wrapper
