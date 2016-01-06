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

import copy
import warnings
import numpy as np
from traits.trait_errors import TraitError

from hyperspy.model import BaseModel, ModelComponents, ModelSpecialSlicers
import hyperspy.drawing.spectrum
from hyperspy.drawing.utils import on_figure_window_close
from hyperspy._signals.eels import Spectrum
from hyperspy.axes import generate_axis
from hyperspy.exceptions import WrongObjectError
from hyperspy.decorators import interactive_range_selector
from hyperspy.axes import AxesManager
from hyperspy.drawing.widgets import (DraggableVerticalLine,
                                      DraggableLabel)
from hyperspy.gui.tools import ComponentFit


class Model1D(BaseModel):
    """Model and data fitting for one dimensional signals.

    A model is constructed as a linear combination of :mod:`components` that
    are added to the model using :meth:`append` or :meth:`extend`. There
    are many predifined components available in the in the :mod:`components`
    module. If needed, new components can be created easily using the code of
    existing components as a template.

    Once defined, the model can be fitted to the data using :meth:`fit` or
    :meth:`multifit`. Once the optimizer reaches the convergence criteria or
    the maximum number of iterations the new value of the component parameters
    are stored in the components.

    It is possible to access the components in the model by their name or by
    the index in the model. An example is given at the end of this docstring.

    Attributes
    ----------

    spectrum : Spectrum instance
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
    as_signal
        Generate a Spectrum instance (possible multidimensional)
        from the model.
    store_current_values
        Store the value of the parameters at the current position.
    fetch_stored_values
        fetch stored values of the parameters.
    update_plot
        Force a plot update. (In most cases the plot should update
        automatically.)
    set_signal_range, remove_signal range, reset_signal_range,
    add signal_range.
        Customize the signal range to fit.
    fit, multifit
        Fit the model to the data at the current position or the
        full dataset.
    save_parameters2file, load_parameters_from_file
        Save/load the parameter values to/from a file.
    plot
        Plot the model and the data.
    enable_plot_components, disable_plot_components
        Plot each component separately. (Use after `plot`.)
    set_current_values_to
        Set the current value of all the parameters of the given component as
        the value for all the dataset.
    export_results
        Save the value of the parameters in separate files.
    plot_results
        Plot the value of all parameters at all positions.
    print_current_values
        Print the value of the parameters at the current position.
    enable_adjust_position, disable_adjust_position
        Enable/disable interactive adjustment of the position of the components
        that have a well defined position. (Use after `plot`).
    fit_component
        Fit just the given component in the given signal range, that can be
        set interactively.
    set_parameters_not_free, set_parameters_free
        Fit the `free` status of several components and parameters at once.
    set_parameters_value
        Set the value of a parameter in components in a model to a specified
        value.
    as_dictionary
        Exports the model to a dictionary that can be saved in a file.

    Examples
    --------
    In the following example we create a histogram from a normal distribution
    and fit it with a gaussian component. It demonstrates how to create
    a model from a :class:`~._signals.spectrum.Spectrum` instance, add
    components to it, adjust the value of the parameters of the components,
    fit the model to the data and access the components in the model.

    >>> s = hs.signals.Spectrum(
            np.random.normal(scale=2, size=10000)).get_histogram()
    >>> g = hs.model.components.Gaussian()
    >>> m = s.create_model()
    >>> m.append(g)
    >>> m.print_current_values()
    Components	Parameter	Value
    Gaussian
                sigma	1.000000
                A	1.000000
                centre	0.000000
    >>> g.centre.value = 3
    >>> m.print_current_values()
    Components	Parameter	Value
    Gaussian
                sigma	1.000000
                A	1.000000
                centre	3.000000
    >>> g.sigma.value
    1.0
    >>> m.fit()
    >>> g.sigma.value
    1.9779042300856682
    >>> m[0].sigma.value
    1.9779042300856682
    >>> m["Gaussian"].centre.value
    -0.072121936813224569

    """
    def __init__(self, spectrum, dictionary=None):
        self.spectrum = spectrum
        self.signal = self.spectrum
        self.axes_manager = self.signal.axes_manager
        self._plot = None
        self._position_widgets = []
        self._adjust_position_all = None
        self._plot_components = False
        self._suspend_update = False
        self._model_line = None
        self._adjust_position_all = None
        self.axis = self.axes_manager.signal_axes[0]
        self.axes_manager.connect(self.fetch_stored_values)
        self.channel_switches = np.array([True] * len(self.axis.axis))
        self.chisq = spectrum._get_navigation_signal()
        self.chisq.change_dtype("float")
        self.chisq.data.fill(np.nan)
        self.chisq.metadata.General.title = self.signal.metadata.General.title + \
            ' chi-squared'
        self.dof = self.chisq._deepcopy_with_new_data(
            np.zeros_like(self.chisq.data, dtype='int'))
        self.dof.metadata.General.title = self.signal.metadata.General.title + \
            ' degrees of freedom'
        self.free_parameters_boundaries = None
        self._low_loss = None
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
            'low_loss': ('sig', None),
            'chisq.data': None,
            'dof.data': None}
        self._slicing_whitelist = {
            'channel_switches': 'isig',
            'low_loss': 'inav',
            'chisq.data': 'inav',
            'dof.data': 'inav'}

    @property
    def spectrum(self):
        return self._spectrum

    @spectrum.setter
    def spectrum(self, value):
        if isinstance(value, Spectrum):
            self._spectrum = value
        else:
            raise WrongObjectError(str(type(value)), 'Spectrum')

    @property
    def low_loss(self):
        return self._low_loss

    @low_loss.setter
    def low_loss(self, value):
        if value is not None:
            if (value.axes_manager.navigation_shape !=
                    self.spectrum.axes_manager.navigation_shape):
                raise ValueError('The low-loss does not have '
                                 'the same navigation dimension as the '
                                 'core-loss')
            self._low_loss = value
            self.set_convolution_axis()
            self.convolved = True
        else:
            self._low_loss = value
            self.convolution_axis = None
            self.convolved = False

    # Extend the list methods to call the _touch when the model is modified

    def set_convolution_axis(self):
        """
        Creates an axis to use to generate the data of the model in the precise
        scale to obtain the correct axis and origin after convolution with the
        lowloss spectrum.
        """
        ll_axis = self.low_loss.axes_manager.signal_axes[0]
        dimension = self.axis.size + ll_axis.size - 1
        step = self.axis.scale
        knot_position = ll_axis.size - ll_axis.value2index(0) - 1
        self.convolution_axis = generate_axis(self.axis.offset, step,
                                              dimension, knot_position)

    def _connect_parameters2update_plot(self):
        if self._plot_active is False:
            return
        for i, component in enumerate(self):
            component.connect(
                self._model_line.update)
            for parameter in component.parameters:
                parameter.connect(self._model_line.update)
        if self._plot_components is True:
            self._connect_component_lines()

    def _disconnect_parameters2update_plot(self):
        if self._model_line is None:
            return
        for component in self:
            component.disconnect(self._model_line.update)
            for parameter in component.parameters:
                parameter.disconnect(self._model_line.update)
        if self._plot_components is True:
            self._disconnect_component_lines()

    def update_plot(self, *args, **kwargs):
        """Update model plot.

        The updating can be suspended using `suspend_update`.

        See Also
        --------
        suspend_update
        resume_update

        """
        if self._plot_active is True and self._suspend_update is False:
            try:
                self._update_model_line()
                for component in [component for component in self if
                                  component.active is True]:
                    self._update_component_line(component)
            except:
                self._disconnect_parameters2update_plot()

    def suspend_update(self):
        """Prevents plot from updating until resume_update() is called

        See Also
        --------
        resume_update
        update_plot
        """
        if self._suspend_update is False:
            self._suspend_update = True
            self._disconnect_parameters2update_plot()
        else:
            warnings.warn("Update already suspended, does nothing.")

    def resume_update(self, update=True):
        """Resumes plot update after suspension by suspend_update()

        Parameters
        ----------
        update : bool, optional
            If True, also updates plot after resuming (default).

        See Also
        --------
        suspend_update
        update_plot
        """
        if self._suspend_update is True:
            self._suspend_update = False
            self._connect_parameters2update_plot()
            if update is True:
                # Ideally, the update flag should in stead work like this:
                # If update is true, update_plot is called if any action
                # would have called it while updating was suspended.
                # However, this is prohibitively difficult to track, so
                # in stead it is simply assume that a change has happened
                # between suspend and resume, and therefore that the plot
                # needs to update. As we do not know what has changed,
                # all components need to update. This can however be
                # suppressed by setting update to false
                self.update_plot()
        else:
            warnings.warn("Update not suspended, nothing to resume.")

    def _update_model_line(self):
        if (self._plot_active is True and
                self._model_line is not None):
            self._model_line.update()

    def __call__(self, non_convolved=False, onlyactive=False):
        """Returns the corresponding model for the current coordinates

        Parameters
        ----------
        non_convolved : bool
            If True it will return the deconvolved model
        only_active : bool
            If True, only the active components will be used to build the
            model.

        cursor: 1 or 2

        Returns
        -------
        numpy array
        """

        if self.convolved is False or non_convolved is True:
            axis = self.axis.axis[self.channel_switches]
            sum_ = np.zeros(len(axis))
            if onlyactive is True:
                for component in self:
                    if component.active:
                        sum_ += component.function(axis)
            else:
                for component in self:
                    sum_ += component.function(axis)
            to_return = sum_

        else:  # convolved
            sum_convolved = np.zeros(len(self.convolution_axis))
            sum_ = np.zeros(len(self.axis.axis))
            for component in self:  # Cut the parameters list
                if onlyactive:
                    if component.active:
                        if component.convolved:
                            sum_convolved += component.function(
                                self.convolution_axis)
                        else:
                            sum_ += component.function(self.axis.axis)
                else:
                    if component.convolved:
                        sum_convolved += component.function(
                            self.convolution_axis)
                    else:
                        sum_ += component.function(self.axis.axis)

            to_return = sum_ + np.convolve(
                self.low_loss(self.axes_manager),
                sum_convolved, mode="valid")
            to_return = to_return[self.channel_switches]
        if self.spectrum.metadata.Signal.binned is True:
            to_return *= self.spectrum.axes_manager[-1].scale
        return to_return

    def _errfunc(self, param, y, weights=None):
        if weights is None:
            weights = 1.
        errfunc = self._model_function(param) - y
        return errfunc * weights

    def _set_signal_range_in_pixels(self, i1=None, i2=None):
        """Use only the selected spectral range in the fitting routine.

        Parameters
        ----------
        i1 : Int
        i2 : Int

        Notes
        -----
        To use the full energy range call the function without arguments.
        """

        self.backup_channel_switches = copy.copy(self.channel_switches)
        self.channel_switches[:] = False
        self.channel_switches[i1:i2] = True
        self.update_plot()

    @interactive_range_selector
    def set_signal_range(self, x1=None, x2=None):
        """Use only the selected spectral range defined in its own units in the
        fitting routine.

        Parameters
        ----------
        E1 : None or float
        E2 : None or float

        Notes
        -----
        To use the full energy range call the function without arguments.
        """
        i1, i2 = self.axis.value_range_to_indices(x1, x2)
        self._set_signal_range_in_pixels(i1, i2)

    def _remove_signal_range_in_pixels(self, i1=None, i2=None):
        """Removes the data in the given range from the data range that
        will be used by the fitting rountine

        Parameters
        ----------
        x1 : None or float
        x2 : None or float
        """
        self.channel_switches[i1:i2] = False
        self.update_plot()

    @interactive_range_selector
    def remove_signal_range(self, x1=None, x2=None):
        """Removes the data in the given range from the data range that
        will be used by the fitting rountine

        Parameters
        ----------
        x1 : None or float
        x2 : None or float

        """
        i1, i2 = self.axis.value_range_to_indices(x1, x2)
        self._remove_signal_range_in_pixels(i1, i2)

    def reset_signal_range(self):
        """Resets the data range"""
        self._set_signal_range_in_pixels()

    def _add_signal_range_in_pixels(self, i1=None, i2=None):
        """Adds the data in the given range from the data range that
        will be used by the fitting rountine

        Parameters
        ----------
        x1 : None or float
        x2 : None or float
        """
        self.channel_switches[i1:i2] = True
        self.update_plot()

    @interactive_range_selector
    def add_signal_range(self, x1=None, x2=None):
        """Adds the data in the given range from the data range that
        will be used by the fitting rountine

        Parameters
        ----------
        x1 : None or float
        x2 : None or float

        """
        i1, i2 = self.axis.value_range_to_indices(x1, x2)
        self._add_signal_range_in_pixels(i1, i2)

    def reset_the_signal_range(self):
        self.channel_switches[:] = True
        self.update_plot()

    def _jacobian(self, param, y, weights=None):
        if weights is None:
            weights = 1.
        if self.convolved is True:
            counter = 0
            grad = np.zeros(len(self.axis.axis))
            for component in self:  # Cut the parameters list
                if component.active:
                    component.fetch_values_from_array(
                        param[
                            counter:counter +
                            component._nfree_param],
                        onlyfree=True)
                    if component.convolved:
                        for parameter in component.free_parameters:
                            par_grad = np.convolve(
                                parameter.grad(self.convolution_axis),
                                self.low_loss(self.axes_manager),
                                mode="valid")
                            if parameter._twins:
                                for par in parameter._twins:
                                    np.add(par_grad, np.convolve(
                                        par.grad(
                                            self.convolution_axis),
                                        self.low_loss(self.axes_manager),
                                        mode="valid"), par_grad)
                            grad = np.vstack((grad, par_grad))
                    else:
                        for parameter in component.free_parameters:
                            par_grad = parameter.grad(self.axis.axis)
                            if parameter._twins:
                                for par in parameter._twins:
                                    np.add(par_grad, par.grad(
                                        self.axis.axis), par_grad)
                            grad = np.vstack((grad, par_grad))
                    counter += component._nfree_param
            to_return = grad[1:, self.channel_switches] * weights
        else:
            axis = self.axis.axis[self.channel_switches]
            counter = 0
            grad = axis
            for component in self:  # Cut the parameters list
                if component.active:
                    component.fetch_values_from_array(
                        param[
                            counter:counter +
                            component._nfree_param],
                        onlyfree=True)
                    for parameter in component.free_parameters:
                        par_grad = parameter.grad(axis)
                        if parameter._twins:
                            for par in parameter._twins:
                                np.add(par_grad, par.grad(
                                    axis), par_grad)
                        grad = np.vstack((grad, par_grad))
                    counter += component._nfree_param
            to_return = grad[1:, :] * weights
        if self.spectrum.metadata.Signal.binned is True:
            to_return *= self.spectrum.axes_manager[-1].scale
        return to_return

    def _function4odr(self, param, x):
        return self._model_function(param)

    def _jacobian4odr(self, param, x):
        return self._jacobian(param, x)

    def _poisson_likelihood_function(self, param, y, weights=None):
        """Returns the likelihood function of the model for the given
        data and parameters
        """
        mf = self._model_function(param)
        with np.errstate(invalid='ignore'):
            return -(y * np.log(mf) - mf).sum()

    def _gradient_ml(self, param, y, weights=None):
        mf = self._model_function(param)
        return -(self._jacobian(param, y) * (y / mf - 1)).sum(1)

    def _gradient_ls(self, param, y, weights=None):
        gls = (2 * self._errfunc(param, y, weights) *
               self._jacobian(param, y)).sum(1)
        return gls

    def plot(self, plot_components=False):
        """Plots the current spectrum to the screen and a map with a
        cursor to explore the SI.

        Parameters
        ----------
        plot_components : bool
            If True, add a line per component to the signal figure.

        """

        # If new coordinates are assigned
        self.spectrum.plot()
        _plot = self.spectrum._plot
        l1 = _plot.signal_plot.ax_lines[0]
        color = l1.line.get_color()
        l1.set_line_properties(color=color, type='scatter')

        l2 = hyperspy.drawing.spectrum.SpectrumLine()
        l2.data_function = self._model2plot
        l2.set_line_properties(color='blue', type='line')
        # Add the line to the figure
        _plot.signal_plot.add_line(l2)
        l2.plot()
        on_figure_window_close(_plot.signal_plot.figure,
                               self._close_plot)

        self._model_line = l2
        self._plot = self.spectrum._plot
        self._connect_parameters2update_plot()
        if plot_components is True:
            self.enable_plot_components()

    @staticmethod
    def _connect_component_line(component):
        if hasattr(component, "_model_plot_line"):
            component.connect(component._model_plot_line.update)
            for parameter in component.parameters:
                parameter.connect(component._model_plot_line.update)

    @staticmethod
    def _disconnect_component_line(component):
        if hasattr(component, "_model_plot_line"):
            component.disconnect(component._model_plot_line.update)
            for parameter in component.parameters:
                parameter.disconnect(component._model_plot_line.update)

    def _connect_component_lines(self):
        for component in [component for component in self if
                          component.active]:
            self._connect_component_line(component)

    def _disconnect_component_lines(self):
        for component in [component for component in self if
                          component.active]:
            self._disconnect_component_line(component)

    def _plot_component(self, component):
        line = hyperspy.drawing.spectrum.SpectrumLine()
        line.data_function = component._component2plot
        # Add the line to the figure
        self._plot.signal_plot.add_line(line)
        line.plot()
        component._model_plot_line = line
        self._connect_component_line(component)

    @staticmethod
    def _update_component_line(component):
        if hasattr(component, "_model_plot_line"):
            component._model_plot_line.update()

    def _disable_plot_component(self, component):
        self._disconnect_component_line(component)
        if hasattr(component, "_model_plot_line"):
            component._model_plot_line.close()
            del component._model_plot_line
        self._plot_components = False

    def _close_plot(self):
        if self._plot_components is True:
            self.disable_plot_components()
        self._disconnect_parameters2update_plot()
        self._model_line = None

    def enable_plot_components(self):
        if self._plot is None or self._plot_components:
            return
        self._plot_components = True
        for component in [component for component in self if
                          component.active]:
            self._plot_component(component)

    def disable_plot_components(self):
        if self._plot is None:
            return
        for component in self:
            self._disable_plot_component(component)
        self._plot_components = False

    def enable_adjust_position(
            self, components=None, fix_them=True, show_label=True):
        """Allow changing the *x* position of component by dragging
        a vertical line that is plotted in the signal model figure

        Parameters
        ----------
        components : {None, list of components}
            If None, the position of all the active components of the
            model that has a well defined *x* position with a value
            in the axis range will get a position adjustment line.
            Otherwise the feature is added only to the given components.
            The components can be specified by name, index or themselves.
        fix_them : bool
            If True the position parameter of the components will be
            temporarily fixed until adjust position is disable.
            This can
            be useful to iteratively adjust the component positions and
            fit the model.
        show_label : bool, optional
            If True, a label showing the component name is added to the
            plot next to the vertical line.

        See also
        --------
        disable_adjust_position

        """
        if (self._plot is None or
                self._plot.is_active() is False):
            self.plot()
        if self._position_widgets:
            self.disable_adjust_position()
        on_figure_window_close(self._plot.signal_plot.figure,
                               self.disable_adjust_position)
        if components:
            components = [self._get_component(x) for x in components]
        else:
            self._adjust_position_all = (fix_them, show_label)

        components = components if components else self
        if not components:
            # The model does not have components so we do nothing
            return
        components = [
            component for component in components if component.active]
        for component in components:
            self._make_position_adjuster(component, fix_them, show_label)

    def _make_position_adjuster(self, component, fix_it, show_label):
        if (component._position is not None and
                not component._position.twin):
            set_value = component._position._set_value
            get_value = component._position._get_value
        else:
            return
        # Create an AxesManager for the widget
        axis_dict = self.axes_manager.signal_axes[0].get_axis_dictionary()
        am = AxesManager([axis_dict, ])
        am._axes[0].navigate = True
        try:
            am._axes[0].value = get_value()
        except TraitError:
            # The value is outside of the axis range
            return
        # Create the vertical line and labels
        if show_label:
            self._position_widgets.extend((
                DraggableVerticalLine(am),
                DraggableLabel(am),))
            # Store the component for bookkeeping, and to reset
            # its twin when disabling adjust position
            self._position_widgets[-2].component = component
            self._position_widgets[-1].component = component
            w = self._position_widgets[-1]
            w.string = component._get_short_description().replace(
                ' component', '')
            w.set_mpl_ax(self._plot.signal_plot.ax)
            self._position_widgets[-2].set_mpl_ax(
                self._plot.signal_plot.ax)
        else:
            self._position_widgets.extend((
                DraggableVerticalLine(am),))
            # Store the component for bookkeeping, and to reset
            # its twin when disabling adjust position
            self._position_widgets[-1].component = component
            self._position_widgets[-1].set_mpl_ax(
                self._plot.signal_plot.ax)
        # Create widget -> parameter connection
        am._axes[0].continuous_value = True
        am._axes[0].on_trait_change(set_value, 'value')
        # Create parameter -> widget connection
        # This is done with a duck typing trick
        # We disguise the AxesManager axis of Parameter by adding
        # the _twin attribute
        am._axes[0]._twins = set()
        component._position.twin = am._axes[0]

    def disable_adjust_position(self):
        """Disables the interactive adjust position feature

        See also
        --------
        enable_adjust_position

        """
        self._adjust_position_all = False
        while self._position_widgets:
            pw = self._position_widgets.pop()
            if hasattr(pw, 'component'):
                pw.component._position.twin = None
                del pw.component
            pw.close()
            del pw

    def fit_component(
            self,
            component,
            signal_range="interactive",
            estimate_parameters=True,
            fit_independent=False,
            only_current=True,
            **kwargs):
        """Fit just the given component in the given signal range.

        This method is useful to obtain starting parameters for the
        components. Any keyword arguments are passed to the fit method.

        Parameters
        ----------
        component : component instance
            The component must be in the model, otherwise an exception
            is raised. The component can be specified by name, index or itself.
        signal_range : {'interactive', (left_value, right_value), None}
            If 'interactive' the signal range is selected using the span
             selector on the spectrum plot. The signal range can also
             be manually specified by passing a tuple of floats. If None
             the current signal range is used.
        estimate_parameters : bool, default True
            If True will check if the component has an
            estimate_parameters function, and use it to estimate the
            parameters in the component.
        fit_independent : bool, default False
            If True, all other components are disabled. If False, all other
            component paramemeters are fixed.

        Examples
        --------
        Signal range set interactivly

        >>> g1 = hs.model.components.Gaussian()
        >>> m.append(g1)
        >>> m.fit_component(g1)

        Signal range set through direct input

        >>> m.fit_component(g1, signal_range=(50,100))
        """
        component = self._get_component(component)
        cf = ComponentFit(self, component, signal_range,
                          estimate_parameters, fit_independent,
                          only_current, **kwargs)
        if signal_range == "interactive":
            cf.edit_traits()
        else:
            cf.apply()
