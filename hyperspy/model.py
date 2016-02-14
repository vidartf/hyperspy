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
import os
import tempfile
import numbers
import numpy as np
import scipy.odr as odr
from scipy.optimize import (leastsq,
                            fmin,
                            fmin_cg,
                            fmin_ncg,
                            fmin_bfgs,
                            fmin_l_bfgs_b,
                            fmin_tnc,
                            fmin_powell)

from hyperspy import messages
from hyperspy.external import progressbar
from hyperspy.defaults_parser import preferences
from hyperspy.external.mpfit.mpfit import mpfit
from hyperspy.component import Component
from hyperspy import components
from hyperspy.signal import Signal
from hyperspy.misc.export_dictionary import (export_to_dictionary,
                                             load_from_dictionary,
                                             parse_flag_string,
                                             reconstruct_object)
from hyperspy.misc.utils import (slugify, shorten_name, stash_active_state,
                                 dummy_context_manager)
from hyperspy.misc.slicing import copy_slice_from_whitelist


class ModelComponents(object):

    """Container for model components.

    Useful to provide tab completion when running in IPython.

    """

    def __init__(self, model):
        self._model = model

    def __repr__(self):
        signature = "%4s | %25s | %25s | %25s"
        ans = signature % ('#',
                           'Attribute Name',
                           'Component Name',
                           'Component Type')
        ans += "\n"
        ans += signature % ('-' * 4, '-' * 25, '-' * 25, '-' * 25)
        if self._model:
            for i, c in enumerate(self._model):
                ans += "\n"
                name_string = c.name
                variable_name = slugify(name_string, valid_variable_name=True)
                component_type = c._id_name

                variable_name = shorten_name(variable_name, 25)
                name_string = shorten_name(name_string, 25)
                component_type = shorten_name(component_type, 25)

                ans += signature % (i,
                                    variable_name,
                                    name_string,
                                    component_type)
        return ans


class BaseModel(list):

    """Model and data fitting tools applicable to signals of both one and two
    dimensions.

    Models of one-dimensional signals should use the :class:`Model1D` and
    models of two-dimensional signals should use the :class:`Model2D`.

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

    signal : Signal instance
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
        Fetch stored values of the parameters.
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

    See also
    --------

    Model1D
    Model2D

    """

    def __hash__(self):
        # This is needed to simulate a hashable object so that PySide does not
        # raise an exception when using windows.connect
        return id(self)

    def store(self, name=None):
        """Stores current model in the original signal

        Parameters
        ----------
            name : {None, str}
                Stored model name. Auto-generated if left empty
        """
        if self.signal is None:
            raise ValueError("Cannot store models with no signal")
        s = self.signal
        s.models.store(self, name)

    def save(self, file_name, name=None, **kwargs):
        """Saves signal and its model to a file

        Parameters
        ----------
            file_name : str
                Name of the file
            name : {None, str}
                Stored model name. Auto-generated if left empty
            **kwargs :
                Other keyword arguments are passed onto `Signal.save()`
        """
        if self.signal is None:
            raise ValueError("Currently cannot store models with no signal")
        else:
            self.store(name)
            self.signal.save(file_name, **kwargs)

    def _load_dictionary(self, dic):
        """Load data from dictionary.

        Parameters
        ----------
        dic : dictionary
            _whitelist : dictionary
                a dictionary with keys used as references of save attributes,
                for more information, see
                :meth:`hyperspy.misc.export_dictionary.load_from_dictionary`
            components : dictionary (optional)
                Dictionary, with information about components of the model (see
                the documentation of component.as_dictionary() method)
            * any field from _whitelist.keys() *
        """

        if 'components' in dic:
            while len(self) != 0:
                self.remove(self[0])
            id_dict = {}

            for comp in dic['components']:
                init_args = {}
                for k, flags_str in comp['_whitelist'].items():
                    if not len(flags_str):
                        continue
                    if 'init' in parse_flag_string(flags_str):
                        init_args[k] = reconstruct_object(flags_str, comp[k])

                self.append(getattr(components, comp['_id_name'])(**init_args))
                id_dict.update(self[-1]._load_dictionary(comp))
            # deal with twins:
            for comp in dic['components']:
                for par in comp['parameters']:
                    for tw in par['_twins']:
                        id_dict[tw].twin = id_dict[par['self']]

        if '_whitelist' in dic:
            load_from_dictionary(self, dic)

    def __repr__(self):
        title = self.signal.metadata.General.title
        class_name = str(self.__class__).split("'")[1].split('.')[-1]

        if len(title):
            return "<%s, title: %s>" % (class_name, self.signal.metadata.General.title)
        else:
            return "<%s>" % class_name

    def _get_component(self, thing):
        if isinstance(thing, int) or isinstance(thing, str):
            thing = self[thing]
        elif not isinstance(thing, Component):
            raise ValueError("Not a component or component id.")
        if thing in self:
            return thing
        else:
            raise ValueError("The component is not in the model.")

    def insert(self, **kwargs):
        raise NotImplementedError

    def append(self, thing):
        """Add component to Model.

        Parameters
        ----------
        thing: `Component` instance.
        """
        if not isinstance(thing, Component):
            raise ValueError(
                "Only `Component` instances can be added to a model")
        # Check if any of the other components in the model has the same name
        if thing in self:
            raise ValueError("Component already in model")
        component_name_list = [component.name for component in self]
        if thing.name:
            name_string = thing.name
        else:
            name_string = thing._id_name

        if name_string in component_name_list:
            temp_name_string = name_string
            index = 0
            while temp_name_string in component_name_list:
                temp_name_string = name_string + "_" + str(index)
                index += 1
            name_string = temp_name_string
        thing.name = name_string

        thing._axes_manager = self.axes_manager
        thing._create_arrays()
        list.append(self, thing)
        thing.model = self
        setattr(self.components, slugify(name_string,
                                         valid_variable_name=True), thing)
        if self._plot_active is True:
            self._connect_parameters2update_plot(components=[thing])
        self.update_plot()

    def extend(self, iterable):
        for object in iterable:
            self.append(object)

    def __delitem__(self, thing):
        things = self.__getitem__(thing)
        if not isinstance(things, list):
            things = [things,]
        for thing in things:
            self.remove(thing)

    def remove(self, thing):
        """Remove component from model.

        Examples
        --------

        >>> s = hs.signals.Spectrum(np.empty(1))
        >>> m = s.create_model()
        >>> g = hs.model.components.Gaussian()
        >>> m.append(g)

        You could remove `g` like this

        >>> m.remove(g)

        Like this:

        >>> m.remove("Gaussian")

        Or like this:

        >>> m.remove(0)

        """
        thing = self._get_component(thing)
        list.remove(self, thing)
        thing.model = None
        if self._plot_active:
            self.update_plot()

    def as_signal(self, component_list=None, out_of_range_to_nan=True,
                  show_progressbar=None):
        """Returns a recreation of the dataset using the model.
        the spectral range that is not fitted is filled with nans.

        Parameters
        ----------
        component_list : list of hyperspy components, optional
            If a list of components is given, only the components given in the
            list is used in making the returned spectrum. The components can
            be specified by name, index or themselves.
        out_of_range_to_nan : bool
            If True the spectral range that is not fitted is filled with nans.
        show_progressbar : None or bool
            If True, display a progress bar. If None the default is set in
            `preferences`.

        Returns
        -------
        spectrum : An instance of the same class as `spectrum`.

        Examples
        --------
        >>> s = hs.signals.Spectrum(np.random.random((10,100)))
        >>> m = s.create_model()
        >>> l1 = hs.model.components.Lorentzian()
        >>> l2 = hs.model.components.Lorentzian()
        >>> m.append(l1)
        >>> m.append(l2)
        >>> s1 = m.as_signal()
        >>> s2 = m.as_signal(component_list=[l1])

        """
        if show_progressbar is None:
            show_progressbar = preferences.General.show_progressbar

        with stash_active_state(self if component_list else []):
            if component_list:
                component_list = [self._get_component(x)
                                  for x in component_list]
                for component_ in self:
                    active = component_ in component_list
                    if component_.active_is_multidimensional:
                        if active:
                            continue    # Keep active_map
                        component_.active_is_multidimensional = False
                    component_.active = active
            data = np.empty(self.signal.data.shape, dtype='float')
            data.fill(np.nan)
            if out_of_range_to_nan is True:
                channel_switches_backup = copy.copy(self.channel_switches)
                self.channel_switches[:] = True
            maxval = self.axes_manager.navigation_size
            pbar = progressbar.progressbar(maxval=maxval,
                                           disabled=not show_progressbar)
            i = 0
            for index in self.axes_manager:
                self.fetch_stored_values(only_fixed=False)
                data[self.axes_manager._getitem_tuple][
                    np.where(self.channel_switches)] = self.__call__(
                    non_convolved=not self.convolved, onlyactive=True).ravel()
                i += 1
                if maxval > 0:
                    pbar.update(i)
            pbar.finish()
            if out_of_range_to_nan is True:
                self.channel_switches[:] = channel_switches_backup
            signal = self.signal.__class__(
                data,
                axes=self.signal.axes_manager._get_axes_dicts())
            signal.metadata.General.title = (
                self.signal.metadata.General.title + " from fitted model")
            signal.metadata.Signal.binned = self.signal.metadata.Signal.binned
        return signal

    @property
    def _plot_active(self):
        if self._plot is not None and self._plot.is_active() is True:
            return True
        else:
            return False

    def _set_p0(self):
        self.p0 = ()
        for component in self:
            if component.active:
                for parameter in component.free_parameters:
                    self.p0 = (self.p0 + (parameter.value,)
                               if parameter._number_of_elements == 1
                               else self.p0 + parameter.value)

    def set_boundaries(self):
        """Generate the boundary list.

        Necessary before fitting with a boundary aware optimizer.

        """
        self.free_parameters_boundaries = []
        for component in self:
            if component.active:
                for param in component.free_parameters:
                    if param._number_of_elements == 1:
                        self.free_parameters_boundaries.append((
                            param._bounds))
                    else:
                        self.free_parameters_boundaries.extend((
                            param._bounds))

    def set_mpfit_parameters_info(self):
        self.mpfit_parinfo = []
        for component in self:
            if component.active:
                for param in component.free_parameters:
                    limited = [False, False]
                    limits = [0, 0]
                    if param.bmin is not None:
                        limited[0] = True
                        limits[0] = param.bmin
                    if param.bmax is not None:
                        limited[1] = True
                        limits[1] = param.bmax
                    if param._number_of_elements == 1:
                        self.mpfit_parinfo.append(
                            {'limited': limited,
                             'limits': limits})
                    else:
                        self.mpfit_parinfo.extend((
                            {'limited': limited,
                             'limits': limits},) * param._number_of_elements)

    def store_current_values(self):
        """ Store the parameters of the current coordinates into the
        parameters array.

        If the parameters array has not being defined yet it creates it filling
        it with the current parameters."""
        for component in self:
            if component.active:
                component.store_current_parameters_in_map()

    def fetch_stored_values(self, only_fixed=False):
        """Fetch the value of the parameters that has been previously stored.

        Parameters
        ----------
        only_fixed : bool
            If True, only the fixed parameters are fetched.

        See Also
        --------
        store_current_values

        """
        cm = (self.suspend_update if self._plot_active
              else dummy_context_manager)
        with cm(update_on_resume=True):
            for component in self:
                component.fetch_stored_values(only_fixed=only_fixed)

    def _fetch_values_from_p0(self, p_std=None):
        """Fetch the parameter values from the output of the optimzer `self.p0`

        Parameters
        ----------
        p_std : array
            array containing the corresponding standard deviatio
            n

        """
        comp_p_std = None
        counter = 0
        for component in self:  # Cut the parameters list
            if component.active is True:
                if p_std is not None:
                    comp_p_std = p_std[
                        counter: counter +
                        component._nfree_param]
                component.fetch_values_from_array(
                    self.p0[counter: counter + component._nfree_param],
                    comp_p_std, onlyfree=True)
                counter += component._nfree_param

    def _model2plot(self, axes_manager, out_of_range2nans=True):
        old_axes_manager = None
        if axes_manager is not self.axes_manager:
            old_axes_manager = self.axes_manager
            self.axes_manager = axes_manager
            self.fetch_stored_values()
        s = self.__call__(non_convolved=False, onlyactive=True)
        if old_axes_manager is not None:
            self.axes_manager = old_axes_manager
            self.fetch_stored_values()
        if out_of_range2nans is True:
            ns = np.empty(self.axis.axis.shape)
            ns.fill(np.nan)
            ns[np.where(self.channel_switches)] = s
            s = ns
        return s

    def _model_function(self, param):
        self.p0 = param
        self._fetch_values_from_p0()
        to_return = self.__call__(non_convolved=False, onlyactive=True)
        return to_return

    def _errfunc2(self, param, y, weights=None):
        if weights is None:
            weights = 1.
        return ((weights * self._errfunc(param, y)) ** 2).sum()

    def _errfunc4mpfit(self, p, fjac=None, x=None, y=None, weights=None):
        if fjac is None:
            errfunc = self._model_function(p) - y
            if weights is not None:
                errfunc *= weights
            status = 0
            return [status, errfunc]
        else:
            return [0, self._jacobian(p, y).T]

    def _calculate_chisq(self):
        if self.signal.metadata.has_item('Signal.Noise_properties.variance'):

            variance = self.signal.metadata.Signal.Noise_properties.variance
            if isinstance(variance, Signal):
                variance = variance.data.__getitem__(
                    self.axes_manager._getitem_tuple)[np.where(
                                                      self.channel_switches)]
        else:
            variance = 1.0
        d = self(onlyactive=True).ravel() - self.signal()[np.where(
            self.channel_switches)]
        d *= d / (1. * variance)  # d = difference^2 / variance.
        self.chisq.data[self.signal.axes_manager.indices[::-1]] = d.sum()

    def _set_current_degrees_of_freedom(self):
        self.dof.data[self.signal.axes_manager.indices[::-1]] = len(self.p0)

    @property
    def red_chisq(self):
        """Reduced chi-squared. Calculated from self.chisq and self.dof
        """
        tmp = self.chisq / (- self.dof + self.channel_switches.sum() - 1)
        tmp.metadata.General.title = self.signal.metadata.General.title + \
            ' reduced chi-squared'
        return tmp

    def fit(self, fitter=None, method='ls', grad=False,
            bounded=False, ext_bounding=False, update_plot=False,
            **kwargs):
        """Fits the model to the experimental data.

        The chi-squared, reduced chi-squared and the degrees of freedom are
        computed automatically when fitting. They are stored as signals, in the
        `chisq`, `red_chisq`  and `dof`. Note that,
        unless ``metadata.Signal.Noise_properties.variance`` contains an
        accurate estimation of the variance of the data, the chi-squared and
        reduced chi-squared cannot be computed correctly. This is also true for
        homocedastic noise.

        Parameters
        ----------
        fitter : {None, "leastsq", "odr", "mpfit", "fmin"}
            The optimizer to perform the fitting. If None the fitter
            defined in `preferences.Model.default_fitter` is used.
            "leastsq" performs least squares using the Levenberg–Marquardt
            algorithm.
            "mpfit"  performs least squares using the Levenberg–Marquardt
            algorithm and, unlike "leastsq", support bounded optimization.
            "fmin" performs curve fitting using a downhill simplex algorithm.
            It is less robust than the Levenberg-Marquardt based optimizers,
            but, at present, it is the only one that support maximum likelihood
            optimization for poissonian noise.
            "odr" performs the optimization using the orthogonal distance
            regression algorithm. It does not support bounds.
            "leastsq", "odr" and "mpfit" can estimate the standard deviation of
            the estimated value of the parameters if the
            "metada.Signal.Noise_properties.variance" attribute is defined.
            Note that if it is not defined the standard deviation is estimated
            using variance equal 1, what, if the noise is heterocedatic, will
            result in a biased estimation of the parameter values and errors.i
            If `variance` is a `Signal` instance of the
            same `navigation_dimension` as the spectrum, and `method` is "ls"
            weighted least squares is performed.
        method : {'ls', 'ml'}
            Choose 'ls' (default) for least squares and 'ml' for poissonian
            maximum-likelihood estimation.  The latter is only available when
            `fitter` is "fmin".
        grad : bool
            If True, the analytical gradient is used if defined to
            speed up the optimization.
        bounded : bool
            If True performs bounded optimization if the fitter
            supports it. Currently only "mpfit" support it.
        update_plot : bool
            If True, the plot is updated during the optimization
            process. It slows down the optimization but it permits
            to visualize the optimization progress.
        ext_bounding : bool
            If True, enforce bounding by keeping the value of the
            parameters constant out of the defined bounding area.

        **kwargs : key word arguments
            Any extra key word argument will be passed to the chosen
            fitter. For more information read the docstring of the optimizer
            of your choice in `scipy.optimize`.

        See Also
        --------
        multifit

        """

        if fitter is None:
            fitter = preferences.Model.default_fitter
        switch_aap = (update_plot != self._plot_active)
        if switch_aap is True and update_plot is False:
            cm = self.suspend_update
        else:
            cm = dummy_context_manager

        with cm(update_on_resume=True):
            self.p_std = None
            self._set_p0()
            if ext_bounding:
                self._enable_ext_bounding()
            if grad is False:
                approx_grad = True
                jacobian = None
                odr_jacobian = None
                grad_ml = None
                grad_ls = None
            else:
                approx_grad = False
                jacobian = self._jacobian
                odr_jacobian = self._jacobian4odr
                grad_ml = self._gradient_ml
                grad_ls = self._gradient_ls

            if bounded is True and fitter not in ("mpfit", "tnc", "l_bfgs_b"):
                raise NotImplementedError(
                    "Bounded optimization is only available for the mpfit "
                    "optimizer.")
            if method == 'ml':
                weights = None
                if fitter != "fmin":
                    raise NotImplementedError(
                        "Maximum likelihood estimation  is only implemented "
                        'for the "fmin" optimizer')
            elif method == "ls":
                metadata = self.signal.metadata
                if "Signal.Noise_properties.variance" not in metadata:
                    variance = 1
                else:
                    variance = metadata.Signal.Noise_properties.variance
                    if isinstance(variance, Signal):
                        if (variance.axes_manager.navigation_shape ==
                                self.signal.axes_manager.navigation_shape):
                            variance = variance.data.__getitem__(
                                self.axes_manager._getitem_tuple)[
                                np.where(self.channel_switches)]
                        else:
                            raise AttributeError(
                                "The `navigation_shape` of the variance "
                                "signals is not equal to the variance shape "
                                "of the signal")
                    elif not isinstance(variance, numbers.Number):
                        raise AttributeError(
                            "Variance must be a number or a `Signal` instance "
                            "but currently it is a %s" % type(variance))

                weights = 1. / np.sqrt(variance)
            else:
                raise ValueError(
                    'method must be "ls" or "ml" but %s given' %
                    method)
            args = (self.signal()[np.where(self.channel_switches)],
                    weights)

            # Least squares "dedicated" fitters
            if fitter == "leastsq":
                output = \
                    leastsq(self._errfunc, self.p0[:], Dfun=jacobian,
                            col_deriv=1, args=args, full_output=True, **kwargs)

                self.p0, pcov = output[0:2]
                signal_len = sum([axis.size
                                  for axis in self.axes_manager.signal_axes])
                if (signal_len > len(self.p0)) and pcov is not None:
                    pcov *= ((self._errfunc(self.p0, *args) ** 2).sum() /
                             (len(args[0]) - len(self.p0)))
                    self.p_std = np.sqrt(np.diag(pcov))
                self.fit_output = output

            elif fitter == "odr":
                modelo = odr.Model(fcn=self._function4odr,
                                   fjacb=odr_jacobian)
                mydata = odr.RealData(
                    self.axis.axis[np.where(self.channel_switches)],
                    self.signal()[np.where(self.channel_switches)],
                    sx=None,
                    sy=(1 / weights if weights is not None else None))
                myodr = odr.ODR(mydata, modelo, beta0=self.p0[:])
                myoutput = myodr.run()
                result = myoutput.beta
                self.p_std = myoutput.sd_beta
                self.p0 = result
                self.fit_output = myoutput

            elif fitter == 'mpfit':
                autoderivative = 1
                if grad is True:
                    autoderivative = 0
                if bounded is True:
                    self.set_mpfit_parameters_info()
                elif bounded is False:
                    self.mpfit_parinfo = None
                m = mpfit(self._errfunc4mpfit, self.p0[:],
                          parinfo=self.mpfit_parinfo, functkw={
                              'y': self.signal()[self.channel_switches],
                              'weights': weights},
                          autoderivative=autoderivative,
                          quiet=1)
                self.p0 = m.params
                if (self.axis.size > len(self.p0)) and m.perror is not None:
                    self.p_std = m.perror * np.sqrt(
                        (self._errfunc(self.p0, *args) ** 2).sum() /
                        (len(args[0]) - len(self.p0)))
                self.fit_output = m
            else:
                # General optimizers (incluiding constrained ones(tnc,l_bfgs_b)
                # Least squares or maximum likelihood
                if method == 'ml':
                    tominimize = self._poisson_likelihood_function
                    fprime = grad_ml
                elif method in ['ls', "wls"]:
                    tominimize = self._errfunc2
                    fprime = grad_ls

                # OPTIMIZERS

                # Simple (don't use gradient)
                if fitter == "fmin":
                    self.p0 = fmin(
                        tominimize, self.p0, args=args, **kwargs)
                elif fitter == "powell":
                    self.p0 = fmin_powell(tominimize, self.p0, args=args,
                                          **kwargs)

                # Make use of the gradient
                elif fitter == "cg":
                    self.p0 = fmin_cg(tominimize, self.p0, fprime=fprime,
                                      args=args, **kwargs)
                elif fitter == "ncg":
                    self.p0 = fmin_ncg(tominimize, self.p0, fprime=fprime,
                                       args=args, **kwargs)
                elif fitter == "bfgs":
                    self.p0 = fmin_bfgs(
                        tominimize, self.p0, fprime=fprime,
                        args=args, **kwargs)

                # Constrainded optimizers

                # Use gradient
                elif fitter == "tnc":
                    if bounded is True:
                        self.set_boundaries()
                    elif bounded is False:
                        self.self.free_parameters_boundaries = None
                    self.p0 = fmin_tnc(
                        tominimize,
                        self.p0,
                        fprime=fprime,
                        args=args,
                        bounds=self.free_parameters_boundaries,
                        approx_grad=approx_grad,
                        **kwargs)[0]
                elif fitter == "l_bfgs_b":
                    if bounded is True:
                        self.set_boundaries()
                    elif bounded is False:
                        self.self.free_parameters_boundaries = None
                    self.p0 = fmin_l_bfgs_b(
                        tominimize, self.p0, fprime=fprime, args=args,
                        bounds=self.free_parameters_boundaries,
                        approx_grad=approx_grad, **kwargs)[0]
                else:
                    print("""
                    The %s optimizer is not available.

                    Available optimizers:
                    Unconstrained:
                    --------------
                    Only least Squares: leastsq and odr
                    General: fmin, powell, cg, ncg, bfgs

                    Cosntrained:
                    ------------
                    tnc and l_bfgs_b
                    """ % fitter)
            if np.iterable(self.p0) == 0:
                self.p0 = (self.p0,)
            self._fetch_values_from_p0(p_std=self.p_std)
            self.store_current_values()
            self._calculate_chisq()
            self._set_current_degrees_of_freedom()
            if ext_bounding is True:
                self._disable_ext_bounding()

    def multifit(self, mask=None, fetch_only_fixed=False,
                 autosave=False, autosave_every=10, show_progressbar=None,
                 interactive_plot=False, **kwargs):
        """Fit the data to the model at all the positions of the
        navigation dimensions.

        Parameters
        ----------

        mask : {None, numpy.array}
            To mask (do not fit) at certain position pass a numpy.array
            of type bool where True indicates that the data will not be
            fitted at the given position.
        fetch_only_fixed : bool
            If True, only the fixed parameters values will be updated
            when changing the positon.
        autosave : bool
            If True, the result of the fit will be saved automatically
            with a frequency defined by autosave_every.
        autosave_every : int
            Save the result of fitting every given number of spectra.

        show_progressbar : None or bool
            If True, display a progress bar. If None the default is set in
            `preferences`.
        interactive_plot : bool
            If True, update the plot for every position as they are processed.
            Note that this slows down the fitting by a lot, but it allows for
            interactive monitoring of the fitting (if in interactive mode).

        **kwargs : key word arguments
            Any extra key word argument will be passed to
            the fit method. See the fit method documentation for
            a list of valid arguments.

        See Also
        --------
        fit

        """
        if show_progressbar is None:
            show_progressbar = preferences.General.show_progressbar

        if autosave is not False:
            fd, autosave_fn = tempfile.mkstemp(
                prefix='hyperspy_autosave-',
                dir='.', suffix='.npz')
            os.close(fd)
            autosave_fn = autosave_fn[:-4]
            messages.information(
                "Autosaving each %s pixels to %s.npz" % (autosave_every,
                                                         autosave_fn))
            messages.information(
                "When multifit finishes its job the file will be deleted")
        if mask is not None and (
            mask.shape != tuple(
                self.axes_manager._navigation_shape_in_array)):
            messages.warning_exit(
                "The mask must be a numpy array of boolen type with "
                " shape: %s" +
                str(self.axes_manager._navigation_shape_in_array))
        masked_elements = 0 if mask is None else mask.sum()
        maxval = self.axes_manager.navigation_size - masked_elements
        if maxval > 0:
            pbar = progressbar.progressbar(maxval=maxval,
                                           disabled=not show_progressbar)
        if 'bounded' in kwargs and kwargs['bounded'] is True:
            if kwargs['fitter'] == 'mpfit':
                self.set_mpfit_parameters_info()
                kwargs['bounded'] = None
            elif kwargs['fitter'] in ("tnc", "l_bfgs_b"):
                self.set_boundaries()
                kwargs['bounded'] = None
            else:
                messages.information(
                    "The chosen fitter does not suppport bounding."
                    "If you require bounding please select one of the "
                    "following fitters instead: mpfit, tnc, l_bfgs_b")
                kwargs['bounded'] = False
        i = 0
        with self.axes_manager.events.indices_changed.suppress_callback(
                self.fetch_stored_values):
            if interactive_plot:
                outer = dummy_context_manager
                inner = self.suspend_update
            else:
                outer = self.suspend_update
                inner = dummy_context_manager
            with outer(update_on_resume=True):
                for index in self.axes_manager:
                    with inner(update_on_resume=True):
                        if mask is None or not mask[index[::-1]]:
                            self.fetch_stored_values(
                                only_fixed=fetch_only_fixed)
                            self.fit(**kwargs)
                            i += 1
                            if maxval > 0:
                                pbar.update(i)
                        if autosave is True and i % autosave_every == 0:
                            self.save_parameters2file(autosave_fn)
                if maxval > 0:
                    pbar.finish()
        if autosave is True:
            messages.information(
                'Deleting the temporary file %s pixels' % (
                    autosave_fn + 'npz'))
            os.remove(autosave_fn + '.npz')

    def save_parameters2file(self, filename):
        """Save the parameters array in binary format.

        The data is saved to a single file in numpy's uncompressed ``.npz``
        format.

        Parameters
        ----------
        filename : str

        See Also
        --------
        load_parameters_from_file, export_results

        Notes
        -----
        This method can be used to save the current state of the model in a way
        that can be loaded back to recreate the it using `load_parameters_from
        file`. Actually, as of HyperSpy 0.8 this is the only way to do so.
        However, this is known to be brittle. For example see
        https://github.com/hyperspy/hyperspy/issues/341.

        """
        kwds = {}
        i = 0
        for component in self:
            cname = component.name.lower().replace(' ', '_')
            for param in component.parameters:
                pname = param.name.lower().replace(' ', '_')
                kwds['%s_%s.%s' % (i, cname, pname)] = param.map
            i += 1
        np.savez(filename, **kwds)

    def load_parameters_from_file(self, filename):
        """Loads the parameters array from  a binary file written with the
        'save_parameters2file' function.

        Parameters
        ---------
        filename : str

        See Also
        --------
        save_parameters2file, export_results

        Notes
        -----
        In combination with `save_parameters2file`, this method can be used to
        recreate a model stored in a file. Actually, before HyperSpy 0.8 this
        is the only way to do so.  However, this is known to be brittle. For
        example see https://github.com/hyperspy/hyperspy/issues/341.

        """
        f = np.load(filename)
        i = 0
        for component in self:  # Cut the parameters list
            cname = component.name.lower().replace(' ', '_')
            for param in component.parameters:
                pname = param.name.lower().replace(' ', '_')
                param.map = f['%s_%s.%s' % (i, cname, pname)]
            i += 1

        self.fetch_stored_values()

    def assign_current_values_to_all(self, components_list=None, mask=None):
        """Set parameter values for all positions to the current ones.

        Parameters
        ----------
        component_list : list of components, optional
            If a list of components is given, the operation will be performed
            only in the value of the parameters of the given components.
            The components can be specified by name, index or themselves.
        mask : boolean numpy array or None, optional
            The operation won't be performed where mask is True.

        """
        if components_list is None:
            components_list = []
            for comp in self:
                if comp.active:
                    components_list.append(comp)
        else:
            components_list = [self._get_component(x) for x in components_list]

        for comp in components_list:
            for parameter in comp.parameters:
                parameter.assign_current_value_to_all(mask=mask)

    def _enable_ext_bounding(self, components=None):
        """
        """
        if components is None:
            components = self
        for component in components:
            for parameter in component.parameters:
                parameter.ext_bounded = True

    def _disable_ext_bounding(self, components=None):
        """
        """
        if components is None:
            components = self
        for component in components:
            for parameter in component.parameters:
                parameter.ext_bounded = False

    def export_results(self, folder=None, format=None, save_std=False,
                       only_free=True, only_active=True):
        """Export the results of the parameters of the model to the desired
        folder.

        Parameters
        ----------
        folder : str or None
            The path to the folder where the file will be saved. If `None` the
            current folder is used by default.
        format : str
            The format to which the data will be exported. It must be the
            extension of any format supported by HyperSpy. If None, the default
            format for exporting as defined in the `Preferences` will be used.
        save_std : bool
            If True, also the standard deviation will be saved.
        only_free : bool
            If True, only the value of the parameters that are free will be
            exported.
        only_active : bool
            If True, only the value of the active parameters will be exported.

        Notes
        -----
        The name of the files will be determined by each the Component and
        each Parameter name attributes. Therefore, it is possible to customise
        the file names modify the name attributes.

        """
        for component in self:
            if only_active is False or component.active:
                component.export(folder=folder, format=format,
                                 save_std=save_std, only_free=only_free)

    def plot_results(self, only_free=True, only_active=True):
        """Plot the value of the parameters of the model

        Parameters
        ----------

        only_free : bool
            If True, only the value of the parameters that are free will be
            plotted.
        only_active : bool
            If True, only the value of the active parameters will be plotted.

        Notes
        -----
        The name of the files will be determined by each the Component and
        each Parameter name attributes. Therefore, it is possible to customise
        the file names modify the name attributes.

        """
        for component in self:
            if only_active is False or component.active:
                component.plot(only_free=only_free)

    def print_current_values(self, only_free=True):
        """Print the value of each parameter of the model.

        Parameters
        ----------
        only_free : bool
            If True, only the value of the parameters that are free will
             be printed.

        """
        print("Components\tParameter\tValue")
        for component in self:
            if component.active:
                if component.name:
                    print(component.name)
                else:
                    print(component._id_name)
                parameters = component.free_parameters if only_free \
                    else component.parameters
                for parameter in parameters:
                    if not hasattr(parameter.value, '__iter__'):
                        print("\t\t%s\t%g" % (
                            parameter.name, parameter.value))

    def set_parameters_not_free(self, component_list=None,
                                parameter_name_list=None):
        """
        Sets the parameters in a component in a model to not free.

        Parameters
        ----------
        component_list : None, or list of hyperspy components, optional
            If None, will apply the function to all components in the model.
            If list of components, will apply the functions to the components
            in the list.  The components can be specified by name, index or
            themselves.
        parameter_name_list : None or list of strings, optional
            If None, will set all the parameters to not free.
            If list of strings, will set all the parameters with the same name
            as the strings in parameter_name_list to not free.

        Examples
        --------
        >>> v1 = hs.model.components.Voigt()
        >>> m.append(v1)
        >>> m.set_parameters_not_free()

        >>> m.set_parameters_not_free(component_list=[v1],
                                      parameter_name_list=['area','centre'])

        See also
        --------
        set_parameters_free
        hyperspy.component.Component.set_parameters_free
        hyperspy.component.Component.set_parameters_not_free
        """

        if not component_list:
            component_list = []
            for _component in self:
                component_list.append(_component)
        else:
            component_list = [self._get_component(x) for x in component_list]

        for _component in component_list:
            _component.set_parameters_not_free(parameter_name_list)

    def set_parameters_free(self, component_list=None,
                            parameter_name_list=None):
        """
        Sets the parameters in a component in a model to free.

        Parameters
        ----------
        component_list : None, or list of hyperspy components, optional
            If None, will apply the function to all components in the model.
            If list of components, will apply the functions to the components
            in the list. The components can be specified by name, index or
            themselves.

        parameter_name_list : None or list of strings, optional
            If None, will set all the parameters to not free.
            If list of strings, will set all the parameters with the same name
            as the strings in parameter_name_list to not free.

        Examples
        --------
        >>> v1 = hs.model.components.Voigt()
        >>> m.append(v1)
        >>> m.set_parameters_free()
        >>> m.set_parameters_free(component_list=[v1],
                                  parameter_name_list=['area','centre'])

        See also
        --------
        set_parameters_not_free
        hyperspy.component.Component.set_parameters_free
        hyperspy.component.Component.set_parameters_not_free
        """

        if not component_list:
            component_list = []
            for _component in self:
                component_list.append(_component)
        else:
            component_list = [self._get_component(x) for x in component_list]

        for _component in component_list:
            _component.set_parameters_free(parameter_name_list)

    def set_parameters_value(
            self,
            parameter_name,
            value,
            component_list=None,
            only_current=False):
        """
        Sets the value of a parameter in components in a model to a specified
        value

        Parameters
        ----------
        parameter_name : string
            Name of the parameter whos value will be changed
        value : number
            The new value of the parameter
        component_list : list of hyperspy components, optional
            A list of components whos parameters will changed. The components
            can be specified by name, index or themselves.

        only_current : bool, default False
            If True, will only change the parameter value at the current
            position in the model.
            If False, will change the parameter value for all the positions.

        Examples
        --------
        >>> v1 = hs.model.components.Voigt()
        >>> v2 = hs.model.components.Voigt()
        >>> m.extend([v1,v2])
        >>> m.set_parameters_value('area', 5)
        >>> m.set_parameters_value('area', 5, component_list=[v1])
        >>> m.set_parameters_value('area', 5, component_list=[v1],
                                   only_current=True)

        """

        if not component_list:
            component_list = []
            for _component in self:
                component_list.append(_component)
        else:
            component_list = [self._get_component(x) for x in component_list]

        for _component in component_list:
            for _parameter in _component.parameters:
                if _parameter.name == parameter_name:
                    if only_current:
                        _parameter.value = value
                        _parameter.store_current_value_in_array()
                    else:
                        _parameter.value = value
                        _parameter.assign_current_value_to_all()

    def as_dictionary(self, fullcopy=True):
        """Returns a dictionary of the model, including all components, degrees
        of freedom (dof) and chi-squared (chisq) with values.

        Parameters
        ----------
        fullcopy : Bool (optional, True)
            Copies of objects are stored, not references. If any found,
            functions will be pickled and signals converted to dictionaries

        Returns
        -------
        dictionary : a complete dictionary of the model, which includes at
        least the following fields:
            components : list
                a list of dictionaries of components, one per
            _whitelist : dictionary
                a dictionary with keys used as references for saved attributes,
                for more information, see
                :meth:`hyperspy.misc.export_dictionary.export_to_dictionary`
            * any field from _whitelist.keys() *
        Examples
        --------
        >>> s = signals.Spectrum(np.random.random((10,100)))
        >>> m = s.create_model()
        >>> l1 = components.Lorentzian()
        >>> l2 = components.Lorentzian()
        >>> m.append(l1)
        >>> m.append(l2)
        >>> d = m.as_dictionary()
        >>> m2 = s.create_model(dictionary=d)

        """
        dic = {'components': [c.as_dictionary(fullcopy) for c in self]}

        export_to_dictionary(self, self._whitelist, dic, fullcopy)

        def remove_empty_numpy_strings(dic):
            for k, v in dic.items():
                if isinstance(v, dict):
                    remove_empty_numpy_strings(v)
                elif isinstance(v, list):
                    for vv in v:
                        if isinstance(vv, dict):
                            remove_empty_numpy_strings(vv)
                        elif isinstance(vv, np.string_) and len(vv) == 0:
                            vv = ''
                elif isinstance(v, np.string_) and len(v) == 0:
                    del dic[k]
                    dic[k] = ''
        remove_empty_numpy_strings(dic)

        return dic

    def set_component_active_value(
            self, value, component_list=None, only_current=False):
        """
        Sets the component 'active' parameter to a specified value

        Parameters
        ----------
        value : bool
            The new value of the 'active' parameter
        component_list : list of hyperspy components, optional
            A list of components whos parameters will changed. The components
            can be specified by name, index or themselves.

        only_current : bool, default False
            If True, will only change the parameter value at the current
            position in the model.
            If False, will change the parameter value for all the positions.

        Examples
        --------
        >>> v1 = hs.model.components.Voigt()
        >>> v2 = hs.model.components.Voigt()
        >>> m.extend([v1,v2])
        >>> m.set_component_active_value(False)
        >>> m.set_component_active_value(True, component_list=[v1])
        >>> m.set_component_active_value(False, component_list=[v1],
                                         only_current=True)

        """

        if not component_list:
            component_list = []
            for _component in self:
                component_list.append(_component)
        else:
            component_list = [self._get_component(x) for x in component_list]

        for _component in component_list:
            _component.active = value
            if _component.active_is_multidimensional:
                if only_current:
                    _component._active_array[
                        self.axes_manager.indices[::-1]] = value
                else:
                    _component._active_array.fill(value)

    def __getitem__(self, value):
        """x.__getitem__(y) <==> x[y]"""
        if isinstance(value, str):
            component_list = []
            for component in self:
                if component.name:
                    if component.name == value:
                        component_list.append(component)
                elif component._id_name == value:
                    component_list.append(component)
            if component_list:
                if len(component_list) == 1:
                    return component_list[0]
                else:
                    raise ValueError(
                        "There are several components with "
                        "the name \"" + str(value) + "\"")
            else:
                raise ValueError(
                    "Component name \"" + str(value) +
                    "\" not found in model")
        else:
            return list.__getitem__(self, value)


class ModelSpecialSlicers(object):

    def __init__(self, model, isNavigation):
        self.isNavigation = isNavigation
        self.model = model

    def __getitem__(self, slices):
        array_slices = self.model.signal._get_array_slices(
            slices,
            self.isNavigation)
        _signal = self.model.signal._slicer(slices, self.isNavigation)
        if _signal.metadata.Signal.signal_type == 'EELS':
            _model = _signal.create_model(
                auto_background=False,
                auto_add_edges=False)
        else:
            _model = _signal.create_model()

        dims = (self.model.axes_manager.navigation_dimension,
                self.model.axes_manager.signal_dimension)
        if self.isNavigation:
            _model.channel_switches[:] = self.model.channel_switches
        else:
            _model.channel_switches[:] = \
                np.atleast_1d(
                    self.model.channel_switches[
                        tuple(array_slices[-dims[1]:])])

        twin_dict = {}
        for comp in self.model:
            init_args = {}
            for k, v in comp._whitelist.items():
                if v is None:
                    continue
                flags_str, value = v
                if 'init' in parse_flag_string(flags_str):
                    init_args[k] = value
            _model.append(getattr(components, comp._id_name)(**init_args))
        copy_slice_from_whitelist(self.model,
                                  _model,
                                  dims,
                                  (slices, array_slices),
                                  self.isNavigation,
                                  )
        for co, cn in zip(self.model, _model):
            copy_slice_from_whitelist(co,
                                      cn,
                                      dims,
                                      (slices, array_slices),
                                      self.isNavigation)
            for po, pn in zip(co.parameters, cn.parameters):
                copy_slice_from_whitelist(po,
                                          pn,
                                          dims,
                                          (slices, array_slices),
                                          self.isNavigation)
                twin_dict[id(po)] = ([id(i) for i in list(po._twins)], pn)

        for k in twin_dict.keys():
            for tw_id in twin_dict[k][0]:
                twin_dict[tw_id][1].twin = twin_dict[k][1]

        _model.chisq.data = _model.chisq.data.copy()
        _model.dof.data = _model.dof.data.copy()
        _model.fetch_stored_values()  # to update and have correct values
        if not self.isNavigation:
            for _ in _model.axes_manager:
                _model._calculate_chisq()

        return _model

# vim: textwidth=80
