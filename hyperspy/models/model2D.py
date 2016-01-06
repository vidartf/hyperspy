# -*- coding: utf-8 -*-
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

from hyperspy.model import BaseModel, ModelComponents, ModelSpecialSlicers
from hyperspy._signals.image import Image
from hyperspy.exceptions import WrongObjectError
from hyperspy.decorators import interactive_range_selector


class Model2D(BaseModel):

    """Model and data fitting for two dimensional signals.

    A model is constructed as a linear combination of :mod:`components2D` that
    are added to the model using :meth:`append` or :meth:`extend`. There
    are many predifined components available in the in the :mod:`components2D`
    module. If needed, new components can be created easily using the code of
    existing components as a template.

    Once defined, the model can be fitted to the data using :meth:`fit` or
    :meth:`multifit`. Once the optimizer reaches the convergence criteria or
    the maximum number of iterations the new value of the component parameters
    are stored in the components.

    It is possible to access the components in the model by their name or by
    the index in the model. An example is given at the end of this docstring.

    Note that methods are not yet defined for plotting 2D models or using
    gradient based optimisation methods - these will be added soon.

    Attributes
    ----------

    image : Image instance
        It contains the data to fit.
    chisq : A Signal of floats
        Chi-squared of the signal (or np.nan if not yet fit)
    dof : A Signal of integers
        Degrees of freedom of the signal (0 if not yet fit)
    red_chisq : Signal instance
        Reduced chi-squared.
    components : `ModelComponents` instance
        The components of the model are attributes of this class. This provides
        a convinient way to access the model components when working in IPython
        as it enables tab completion.

    Methods
    -------

    append
        Append one component to the model.
    extend
        Append multiple components to the model.
    remove
        Remove component from model.
    fit, multifit
        Fit the model to the data at the current position or the full dataset.

    See also
    --------
    Base Model
    Model1D

    Example
    -------



    """

    def __init__(self, image, dictionary=None):
        self.image = image
        self.signal = self.image
        self.axes_manager = self.signal.axes_manager
        self._plot = None
        self._position_widgets = []
        self._adjust_position_all = None
        self._plot_components = False
        self._suspend_update = False
        self._model_line = None
        self._adjust_position_all = None
        self.xaxis, self.yaxis = np.meshgrid(
            self.axes_manager.signal_axes[0].axis,
            self.axes_manager.signal_axes[1].axis)
        self.axes_manager.connect(self.fetch_stored_values)
        self.channel_switches = np.ones(self.xaxis.shape, dtype=bool)
        self.chisq = image._get_navigation_signal()
        self.chisq.change_dtype("float")
        self.chisq.data.fill(np.nan)
        self.chisq.metadata.General.title = self.signal.metadata.General.title + \
            ' chi-squared'
        self.dof = self.chisq._deepcopy_with_new_data(
            np.zeros_like(self.chisq.data, dtype='int'))
        self.dof.metadata.General.title = self.signal.metadata.General.title + \
            ' degrees of freedom'
        self.free_parameters_boundaries = None
        self.convolved = False
        self.components = ModelComponents(self)
        if dictionary is not None:
            self._load_dictionary(dictionary)
        self.inav = ModelSpecialSlicers(self, True)
        self.isig = ModelSpecialSlicers(self, False)
        self._whitelist = {
            'channel_switches': None,
            'convolved': None,
            'free_parameters_boundaries': None,
            'chisq.data': None,
            'dof.data': None}
        self._slicing_whitelist = {
            'channel_switches': 'isig',
            'chisq.data': 'inav',
            'dof.data': 'inav'}

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, value):
        if isinstance(value, Image):
            self._image = value
        else:
            raise WrongObjectError(str(type(value)), 'Image')

    def __call__(self, non_convolved=True, onlyactive=False):
        """Returns the corresponding 2D model for the current coordinates

        Parameters
        ----------
        only_active : bool
            If true, only the active components will be used to build the model.

        Returns
        -------
        numpy array
        """

        sum_ = np.zeros_like(self.xaxis)
        if onlyactive is True:
            for component in self:  # Cut the parameters list
                if component.active:
                    np.add(sum_, component.function(self.xaxis, self.yaxis),
                           sum_)
        else:
            for component in self:  # Cut the parameters list
                np.add(sum_, component.function(self.xaxis, self.yaxis),
                       sum_)
        return sum_

    def _errfunc(self, param, y, weights=None):
        if weights is None:
            weights = 1.
        errfunc = self._model_function(param).ravel() - y
        return errfunc * weights

    # TODO: The methods below are implemented only for Model1D and should be
    # added eventually also for Model2D. Probably there are smarter ways to do
    # it than redefining every method, but it is structured this way now to make
    # clear what is and isn't available
    def _connect_parameters2update_plot(self):
        raise NotImplementedError

    def _disconnect_parameters2update_plot(self):
        raise NotImplementedError

    def update_plot(self, *args, **kwargs):
        raise NotImplementedError

    def suspend_update(self):
        raise NotImplementedError

    def resume_update(self, update=True):
        raise NotImplementedError

    def _update_model_line(self):
        raise NotImplementedError

    def _set_signal_range_in_pixels(self, i1=None, i2=None):
        raise NotImplementedError

    @interactive_range_selector
    def set_signal_range(self, x1=None, x2=None):
        raise NotImplementedError

    def _remove_signal_range_in_pixels(self, i1=None, i2=None):
        raise NotImplementedError

    @interactive_range_selector
    def remove_signal_range(self, x1=None, x2=None):
        raise NotImplementedError

    def reset_signal_range(self):
        raise NotImplementedError

    def _add_signal_range_in_pixels(self, i1=None, i2=None):
        raise NotImplementedError

    @interactive_range_selector
    def add_signal_range(self, x1=None, x2=None):
        raise NotImplementedError

    def reset_the_signal_range(self):
        raise NotImplementedError

    def _jacobian(self, param, y, weights=None):
        raise NotImplementedError

    def _function4odr(self, param, x):
        raise NotImplementedError

    def _jacobian4odr(self, param, x):
        raise NotImplementedError

    def _poisson_likelihood_function(self, param, y, weights=None):
        raise NotImplementedError

    def _gradient_ml(self, param, y, weights=None):
        raise NotImplementedError

    def _gradient_ls(self, param, y, weights=None):
        raise NotImplementedError

    def plot(self, plot_components=False):
        raise NotImplementedError

    @staticmethod
    def _connect_component_line(component):
        raise NotImplementedError

    @staticmethod
    def _disconnect_component_line(component):
        raise NotImplementedError

    def _connect_component_lines(self):
        raise NotImplementedError

    def _disconnect_component_lines(self):
        raise NotImplementedError

    def _plot_component(self, component):
        raise NotImplementedError

    @staticmethod
    def _update_component_line(component):
        raise NotImplementedError

    def _disable_plot_component(self, component):
        raise NotImplementedError

    def _close_plot(self):
        raise NotImplementedError

    def enable_plot_components(self):
        raise NotImplementedError

    def disable_plot_components(self):
        raise NotImplementedError

    def enable_adjust_position(
            self, components=None, fix_them=True, show_label=True):
        raise NotImplementedError

    def disable_adjust_position(self):
        raise NotImplementedError
