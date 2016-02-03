import numpy as np
import nose.tools as nt
import mock

import hyperspy.api as hs
from hyperspy.misc.utils import slugify


class TestModelJacobians:

    def setUp(self):
        s = hs.signals.Spectrum(np.zeros(1))
        m = s.create_model()
        self.low_loss = 7.
        self.weights = 0.3
        m.axis.axis = np.array([1, 0])
        m.channel_switches = np.array([0, 1], dtype=bool)
        m.append(hs.model.components.Gaussian())
        m[0].A.value = 1
        m[0].centre.value = 2.
        m[0].sigma.twin = m[0].centre
        m._low_loss = mock.MagicMock()
        m.low_loss.return_value = self.low_loss
        self.model = m
        m.convolution_axis = np.zeros(2)

    def test_jacobian_not_convolved(self):
        m = self.model
        m.convolved = False
        jac = m._jacobian((1, 2, 3), None, weights=self.weights)
        np.testing.assert_array_almost_equal(jac.squeeze(), self.weights *
                                             np.array([m[0].A.grad(0),
                                                       m[0].sigma.grad(0) +
                                                       m[0].centre.grad(0)]))
        nt.assert_equal(m[0].A.value, 1)
        nt.assert_equal(m[0].centre.value, 2)
        nt.assert_equal(m[0].sigma.value, 2)

    def test_jacobian_convolved(self):
        m = self.model
        m.convolved = True
        m.append(hs.model.components.Gaussian())
        m[0].convolved = False
        m[1].convolved = True
        jac = m._jacobian((1, 2, 3, 4, 5), None, weights=self.weights)
        np.testing.assert_array_almost_equal(jac.squeeze(), self.weights *
                                             np.array([m[0].A.grad(0),
                                                       m[0].sigma.grad(0) +
                                                       m[0].centre.grad(0),
                                                       m[1].A.grad(0) *
                                                       self.low_loss,
                                                       m[1].centre.grad(0) *
                                                       self.low_loss,
                                                       m[1].sigma.grad(0) *
                                                       self.low_loss,
                                                       ]))
        nt.assert_equal(m[0].A.value, 1)
        nt.assert_equal(m[0].centre.value, 2)
        nt.assert_equal(m[0].sigma.value, 2)
        nt.assert_equal(m[1].A.value, 3)
        nt.assert_equal(m[1].centre.value, 4)
        nt.assert_equal(m[1].sigma.value, 5)


class TestModelCallMethod:

    def setUp(self):
        s = hs.signals.Spectrum(np.empty(1))
        m = s.create_model()
        m.append(hs.model.components.Gaussian())
        m.append(hs.model.components.Gaussian())
        self.model = m

    def test_call_method_no_convolutions(self):
        m = self.model
        m.convolved = False

        m[1].active = False
        r1 = m()
        r2 = m(onlyactive=True)
        np.testing.assert_almost_equal(m[0].function(0) * 2, r1)
        np.testing.assert_almost_equal(m[0].function(0), r2)

        m.convolved = True
        r1 = m(non_convolved=True)
        r2 = m(non_convolved=True, onlyactive=True)
        np.testing.assert_almost_equal(m[0].function(0) * 2, r1)
        np.testing.assert_almost_equal(m[0].function(0), r2)

    def test_call_method_with_convolutions(self):
        m = self.model
        m._low_loss = mock.MagicMock()
        m.low_loss.return_value = 0.3
        m.convolved = True

        m.append(hs.model.components.Gaussian())
        m[1].active = False
        m[0].convolved = True
        m[1].convolved = False
        m[2].convolved = False
        m.convolution_axis = np.array([0., ])

        r1 = m()
        r2 = m(onlyactive=True)
        np.testing.assert_almost_equal(m[0].function(0) * 2.3, r1)
        np.testing.assert_almost_equal(m[0].function(0) * 1.3, r2)

    def test_call_method_binned(self):
        m = self.model
        m.convolved = False
        m.remove(1)
        m.spectrum.metadata.Signal.binned = True
        m.spectrum.axes_manager[-1].scale = 0.3
        r1 = m()
        np.testing.assert_almost_equal(m[0].function(0) * 0.3, r1)


class TestModelPlotCall:

    def setUp(self):
        s = hs.signals.Spectrum(np.empty(1))
        m = s.create_model()
        m.__call__ = mock.MagicMock()
        m.__call__.return_value = np.array([0.5, 0.25])
        m.axis = mock.MagicMock()
        m.fetch_stored_values = mock.MagicMock()
        m.channel_switches = np.array([0, 1, 1, 0, 0], dtype=bool)
        self.model = m

    def test_model2plot_own_am(self):
        m = self.model
        m.axis.axis.shape = (5,)
        res = m._model2plot(m.axes_manager)
        np.testing.assert_array_equal(
            res, np.array([np.nan, 0.5, 0.25, np.nan, np.nan]))
        nt.assert_true(m.__call__.called)
        nt.assert_dict_equal(
            m.__call__.call_args[1], {
                'non_convolved': False, 'onlyactive': True})
        nt.assert_false(m.fetch_stored_values.called)

    def test_model2plot_other_am(self):
        m = self.model
        res = m._model2plot(m.axes_manager.deepcopy(), out_of_range2nans=False)
        np.testing.assert_array_equal(res, np.array([0.5, 0.25]))
        nt.assert_true(m.__call__.called)
        nt.assert_dict_equal(
            m.__call__.call_args[1], {
                'non_convolved': False, 'onlyactive': True})
        nt.assert_equal(2, m.fetch_stored_values.call_count)


class TestModelSettingPZero:

    def setUp(self):
        s = hs.signals.Spectrum(np.empty(1))
        m = s.create_model()
        m.append(hs.model.components.Gaussian())

        m[0].A.value = 1.1
        m[0].centre._number_of_elements = 2
        m[0].centre.value = (2.2, 3.3)
        m[0].sigma.value = 4.4
        m[0].sigma.free = False

        m[0].A._bounds = (0.1, 0.11)
        m[0].centre._bounds = ((0.2, 0.21), (0.3, 0.31))
        m[0].sigma._bounds = (0.4, 0.41)

        self.model = m

    def test_setting_p0(self):
        m = self.model
        m.append(hs.model.components.Gaussian())
        m[-1].active = False
        m.p0 = None
        m._set_p0()
        nt.assert_equal(m.p0, (1.1, 2.2, 3.3))

    def test_fetching_from_p0(self):
        m = self.model

        m.append(hs.model.components.Gaussian())
        m[-1].active = False
        m[-1].A.value = 100
        m[-1].sigma.value = 200
        m[-1].centre.value = 300

        m.p0 = (1.2, 2.3, 3.4, 5.6, 6.7, 7.8)
        m._fetch_values_from_p0()
        nt.assert_equal(m[0].A.value, 1.2)
        nt.assert_equal(m[0].centre.value, (2.3, 3.4))
        nt.assert_equal(m[0].sigma.value, 4.4)
        nt.assert_equal(m[1].A.value, 100)
        nt.assert_equal(m[1].sigma.value, 200)
        nt.assert_equal(m[1].centre.value, 300)

    def test_setting_boundaries(self):
        m = self.model
        m.append(hs.model.components.Gaussian())
        m[-1].active = False
        m.set_boundaries()
        nt.assert_equal(m.free_parameters_boundaries,
                        [(0.1, 0.11), (0.2, 0.21), (0.3, 0.31)])

    def test_setting_mpfit_parameters_info(self):
        m = self.model
        m[0].A.bmax = None
        m[0].centre.bmin = None
        m[0].centre.bmax = 0.31
        m.append(hs.model.components.Gaussian())
        m[-1].active = False
        m.set_mpfit_parameters_info()
        nt.assert_equal(m.mpfit_parinfo,
                        [{'limited': [True, False],
                          'limits': [0.1, 0]},
                         {'limited': [False, True],
                          'limits': [0, 0.31]},
                         {'limited': [False, True],
                          'limits': [0, 0.31]},
                         ])


class TestModel1D:

    def setUp(self):
        s = hs.signals.Spectrum(np.empty(1))
        m = s.create_model()
        self.model = m

    def test_errfunc(self):
        m = self.model
        m._model_function = mock.MagicMock()
        m._model_function.return_value = 3.
        np.testing.assert_equal(m._errfunc(None, 1., None), 2.)
        np.testing.assert_equal(m._errfunc(None, 1., 0.3), 0.6)

    def test_errfunc2(self):
        m = self.model
        m._model_function = mock.MagicMock()
        m._model_function.return_value = 3. * np.ones(2)
        np.testing.assert_equal(m._errfunc2(None, np.ones(2), None), 2 * 4.)
        np.testing.assert_equal(m._errfunc2(None, np.ones(2), 0.3), 2 * 0.36)

    def test_gradient_ls(self):
        m = self.model
        m._errfunc = mock.MagicMock()
        m._errfunc.return_value = 0.1
        m._jacobian = mock.MagicMock()
        m._jacobian.return_value = np.ones((1, 2)) * 7.
        np.testing.assert_equal(m._gradient_ls(None, None), 2 * 0.1 * 7 * 2)

    def test_gradient_ml(self):
        m = self.model
        m._model_function = mock.MagicMock()
        m._model_function.return_value = 3. * np.ones(2)
        m._jacobian = mock.MagicMock()
        m._jacobian.return_value = np.ones((1, 2)) * 7.
        np.testing.assert_equal(
            m._gradient_ml(None, 1.2), -2 * 7 * (1.2 / 3 - 1))

    def test_model_function(self):
        m = self.model
        m.append(hs.model.components.Gaussian())
        m[0].A.value = 1.3
        m[0].centre.value = 0.003
        m[0].sigma.value = 0.1
        param = (100, 0.1, 0.2)
        np.testing.assert_array_almost_equal(176.03266338,
                                             m._model_function(param))
        nt.assert_equal(m[0].A.value, 100)
        nt.assert_equal(m[0].centre.value, 0.1)
        nt.assert_equal(m[0].sigma.value, 0.2)

    @nt.raises(ValueError)
    def test_append_existing_component(self):
        g = hs.model.components.Gaussian()
        m = self.model
        m.append(g)
        m.append(g)

    def test_append_component(self):
        g = hs.model.components.Gaussian()
        m = self.model
        m.append(g)
        nt.assert_in(g, m)
        nt.assert_is(g.model, m)
        nt.assert_is(g._axes_manager, m.axes_manager)
        nt.assert_true(all([hasattr(p, 'map') for p in g.parameters]))

    def test_calculating_convolution_axis(self):
        m = self.model
        # setup
        m.axis.offset = 10
        m.axis.size = 10
        ll_axis = mock.MagicMock()
        ll_axis.size = 7
        ll_axis.value2index.return_value = 3
        m._low_loss = mock.MagicMock()
        m.low_loss.axes_manager.signal_axes = [ll_axis, ]

        # calculation
        m.set_convolution_axis()

        # tests
        np.testing.assert_array_equal(m.convolution_axis, np.arange(7, 23))
        nt.assert_equal(ll_axis.value2index.call_args[0][0], 0)

    def test_access_component_by_name(self):
        m = self.model
        g1 = hs.model.components.Gaussian()
        g2 = hs.model.components.Gaussian()
        g2.name = "test"
        m.extend((g1, g2))
        nt.assert_is(m["test"], g2)

    def test_access_component_by_index(self):
        m = self.model
        g1 = hs.model.components.Gaussian()
        g2 = hs.model.components.Gaussian()
        g2.name = "test"
        m.extend((g1, g2))
        nt.assert_is(m[1], g2)

    def test_component_name_when_append(self):
        m = self.model
        gs = [
            hs.model.components.Gaussian(),
            hs.model.components.Gaussian(),
            hs.model.components.Gaussian()]
        m.extend(gs)
        nt.assert_is(m['Gaussian'], gs[0])
        nt.assert_is(m['Gaussian_0'], gs[1])
        nt.assert_is(m['Gaussian_1'], gs[2])

    @nt.raises(ValueError)
    def test_several_component_with_same_name(self):
        m = self.model
        gs = [
            hs.model.components.Gaussian(),
            hs.model.components.Gaussian(),
            hs.model.components.Gaussian()]
        m.extend(gs)
        m[0]._name = "hs.model.components.Gaussian"
        m[1]._name = "hs.model.components.Gaussian"
        m[2]._name = "hs.model.components.Gaussian"
        m['Gaussian']

    @nt.raises(ValueError)
    def test_no_component_with_that_name(self):
        m = self.model
        m['Voigt']

    @nt.raises(ValueError)
    def test_component_already_in_model(self):
        m = self.model
        g1 = hs.model.components.Gaussian()
        m.extend((g1, g1))

    def test_remove_component(self):
        m = self.model
        g1 = hs.model.components.Gaussian()
        m.append(g1)
        m.remove(g1)
        nt.assert_equal(len(m), 0)

    def test_remove_component_by_index(self):
        m = self.model
        g1 = hs.model.components.Gaussian()
        m.append(g1)
        m.remove(0)
        nt.assert_equal(len(m), 0)

    def test_remove_component_by_name(self):
        m = self.model
        g1 = hs.model.components.Gaussian()
        m.append(g1)
        m.remove(g1.name)
        nt.assert_equal(len(m), 0)

    def test_delete_component_by_index(self):
        m = self.model
        g1 = hs.model.components.Gaussian()
        m.append(g1)
        del m[0]
        nt.assert_not_in(g1, m)

    def test_delete_component_by_name(self):
        m = self.model
        g1 = hs.model.components.Gaussian()
        m.append(g1)
        del m[g1.name]
        nt.assert_not_in(g1, m)

    def test_delete_slice(self):
        m = self.model
        g1 = hs.model.components.Gaussian()
        g2 = hs.model.components.Gaussian()
        g3 = hs.model.components.Gaussian()
        m.extend([g1, g2, g3])
        del m[:2]
        nt.assert_not_in(g1, m)
        nt.assert_not_in(g2, m)
        nt.assert_in(g3, m)

    def test_get_component_by_name(self):
        m = self.model
        g1 = hs.model.components.Gaussian()
        g2 = hs.model.components.Gaussian()
        g2.name = "test"
        m.extend((g1, g2))
        nt.assert_is(m._get_component("test"), g2)

    def test_get_component_by_index(self):
        m = self.model
        g1 = hs.model.components.Gaussian()
        g2 = hs.model.components.Gaussian()
        g2.name = "test"
        m.extend((g1, g2))
        nt.assert_is(m._get_component(1), g2)

    def test_get_component_by_component(self):
        m = self.model
        g1 = hs.model.components.Gaussian()
        g2 = hs.model.components.Gaussian()
        g2.name = "test"
        m.extend((g1, g2))
        nt.assert_is(m._get_component(g2), g2)

    @nt.raises(ValueError)
    def test_get_component_wrong(self):
        m = self.model
        g1 = hs.model.components.Gaussian()
        g2 = hs.model.components.Gaussian()
        g2.name = "test"
        m.extend((g1, g2))
        m._get_component(1.2)

    def test_components_class_default(self):
        m = self.model
        g1 = hs.model.components.Gaussian()
        m.append(g1)
        nt.assert_is(getattr(m.components, g1.name), g1)

    def test_components_class_change_name(self):
        m = self.model
        g1 = hs.model.components.Gaussian()
        m.append(g1)
        g1.name = "test"
        nt.assert_is(getattr(m.components, g1.name), g1)

    @nt.raises(AttributeError)
    def test_components_class_change_name_del_default(self):
        m = self.model
        g1 = hs.model.components.Gaussian()
        m.append(g1)
        g1.name = "test"
        getattr(m.components, "Gaussian")

    def test_components_class_change_invalid_name(self):
        m = self.model
        g1 = hs.model.components.Gaussian()
        m.append(g1)
        g1.name = "1, Test This!"
        nt.assert_is(
            getattr(m.components,
                    slugify(g1.name, valid_variable_name=True)), g1)

    @nt.raises(AttributeError)
    def test_components_class_change_name_del_default(self):
        m = self.model
        g1 = hs.model.components.Gaussian()
        m.append(g1)
        invalid_name = "1, Test This!"
        g1.name = invalid_name
        g1.name = "test"
        getattr(m.components, slugify(invalid_name))


class TestModel2D:

    def setUp(self):
        g = hs.model.components.Gaussian2D(
            centre_x=-5.,
            centre_y=-5.,
            sigma_x=1.,
            sigma_y=2.)
        x = np.arange(-10, 10, 0.01)
        y = np.arange(-10, 10, 0.01)
        X, Y = np.meshgrid(x, y)
        im = hs.signals.Image(g.function(X, Y))
        im.axes_manager[0].scale = 0.01
        im.axes_manager[0].offset = -10
        im.axes_manager[1].scale = 0.01
        im.axes_manager[1].offset = -10
        self.im = im

    def test_fitting(self):
        im = self.im
        m = im.create_model()
        gt = hs.model.components.Gaussian2D(centre_x=-4.5,
                                            centre_y=-4.5,
                                            sigma_x=0.5,
                                            sigma_y=1.5)
        m.append(gt)
        m.fit()
        nt.assert_almost_equal(gt.centre_x.value, -5.)
        nt.assert_almost_equal(gt.centre_y.value, -5.)
        nt.assert_almost_equal(gt.sigma_x.value, 1.)
        nt.assert_almost_equal(gt.sigma_y.value, 2.)


class TestModelFitBinned:

    def setUp(self):
        np.random.seed(1)
        s = hs.signals.Spectrum(
            np.random.normal(
                scale=2,
                size=10000)).get_histogram()
        s.metadata.Signal.binned = True
        g = hs.model.components.Gaussian()
        m = s.create_model()
        m.append(g)
        g.sigma.value = 1
        g.centre.value = 0.5
        g.A.value = 1e3
        self.m = m

    def test_fit_fmin_leastsq(self):
        self.m.fit(fitter="fmin", method="ls")
        nt.assert_almost_equal(self.m[0].A.value, 9976.14519369)
        nt.assert_almost_equal(self.m[0].centre.value, -0.110610743285)
        nt.assert_almost_equal(self.m[0].sigma.value, 1.98380705455)

    def test_fit_fmin_ml(self):
        self.m.fit(fitter="fmin", method="ml")
        nt.assert_almost_equal(self.m[0].A.value, 10001.39613936,
                               places=3)
        nt.assert_almost_equal(self.m[0].centre.value, -0.104151206314,
                               places=6)
        nt.assert_almost_equal(self.m[0].sigma.value, 2.00053642434)

    def test_fit_leastsq(self):
        self.m.fit(fitter="leastsq")
        nt.assert_almost_equal(self.m[0].A.value, 9976.14526082, 1)
        nt.assert_almost_equal(self.m[0].centre.value, -0.110610727064)
        nt.assert_almost_equal(self.m[0].sigma.value, 1.98380707571, 5)

    def test_fit_mpfit(self):
        self.m.fit(fitter="mpfit")
        nt.assert_almost_equal(self.m[0].A.value, 9976.14526286, 5)
        nt.assert_almost_equal(self.m[0].centre.value, -0.110610718444)
        nt.assert_almost_equal(self.m[0].sigma.value, 1.98380707614)

    def test_fit_odr(self):
        self.m.fit(fitter="odr")
        nt.assert_almost_equal(self.m[0].A.value, 9976.14531979, 3)
        nt.assert_almost_equal(self.m[0].centre.value, -0.110610724054)
        nt.assert_almost_equal(self.m[0].sigma.value, 1.98380709939)

    def test_fit_leastsq_grad(self):
        self.m.fit(fitter="leastsq", grad=True)
        nt.assert_almost_equal(self.m[0].A.value, 9976.14526084)
        nt.assert_almost_equal(self.m[0].centre.value, -0.11061073306)
        nt.assert_almost_equal(self.m[0].sigma.value, 1.98380707552)

    def test_fit_mpfit_grad(self):
        self.m.fit(fitter="mpfit", grad=True)
        nt.assert_almost_equal(self.m[0].A.value, 9976.14526084)
        nt.assert_almost_equal(self.m[0].centre.value, -0.11061073306)
        nt.assert_almost_equal(self.m[0].sigma.value, 1.98380707552)

    def test_fit_odr_grad(self):
        self.m.fit(fitter="odr", grad=True)
        nt.assert_almost_equal(self.m[0].A.value, 9976.14531979, 3)
        nt.assert_almost_equal(self.m[0].centre.value, -0.110610724054)
        nt.assert_almost_equal(self.m[0].sigma.value, 1.98380709939)

    def test_fit_bounded(self):
        self.m[0].centre.bmin = 0.5
        self.m[0].bounded = True
        self.m.fit(fitter="mpfit", bounded=True)
        nt.assert_almost_equal(self.m[0].A.value, 9991.65422046, 4)
        nt.assert_almost_equal(self.m[0].centre.value, 0.5)
        nt.assert_almost_equal(self.m[0].sigma.value, 2.08398236966)

    @nt.raises(ValueError)
    def test_wrong_method(self):
        self.m.fit(method="dummy")


class TestModelWeighted:

    def setUp(self):
        np.random.seed(1)
        s = hs.signals.SpectrumSimulation(np.arange(10, 100, 0.1))
        s.metadata.set_item("Signal.Noise_properties.variance",
                            hs.signals.Spectrum(np.arange(10, 100, 0.01)))
        s.axes_manager[0].scale = 0.1
        s.axes_manager[0].offset = 10
        s.add_poissonian_noise()
        m = s.create_model()
        m.append(hs.model.components.Polynomial(1))
        self.m = m

    def test_fit_leastsq_binned(self):
        self.m.spectrum.metadata.Signal.binned = True
        self.m.fit(fitter="leastsq", method="ls")
        for result, expected in zip(self.m[0].coefficients.value,
                                    (9.9165596693502778, 1.6628238107916631)):
            nt.assert_almost_equal(result, expected, places=5)

    def test_fit_odr_binned(self):
        self.m.spectrum.metadata.Signal.binned = True
        self.m.fit(fitter="odr", method="ls")
        for result, expected in zip(self.m[0].coefficients.value,
                                    (9.9165596548961972, 1.6628247412317521)):
            nt.assert_almost_equal(result, expected, places=5)

    def test_fit_mpfit_binned(self):
        self.m.spectrum.metadata.Signal.binned = True
        self.m.fit(fitter="mpfit", method="ls")
        for result, expected in zip(self.m[0].coefficients.value,
                                    (9.9165596607108739, 1.6628243846485873)):
            nt.assert_almost_equal(result, expected, places=5)

    def test_fit_fmin_binned(self):
        self.m.spectrum.metadata.Signal.binned = True
        self.m.fit(
            fitter="fmin",
            method="ls",
        )
        for result, expected in zip(self.m[0].coefficients.value,
                                    (9.9137288425667442, 1.8446013472266145)):
            nt.assert_almost_equal(result, expected, places=5)

    def test_fit_leastsq_unbinned(self):
        self.m.spectrum.metadata.Signal.binned = False
        self.m.fit(fitter="leastsq", method="ls")
        for result, expected in zip(
                self.m[0].coefficients.value,
                (0.99165596391487121, 0.16628254242532492)):
            nt.assert_almost_equal(result, expected, places=5)

    def test_fit_odr_unbinned(self):
        self.m.spectrum.metadata.Signal.binned = False
        self.m.fit(fitter="odr", method="ls")
        for result, expected in zip(
                self.m[0].coefficients.value,
                (0.99165596548961943, 0.16628247412317315)):
            nt.assert_almost_equal(result, expected, places=5)

    def test_fit_mpfit_unbinned(self):
        self.m.spectrum.metadata.Signal.binned = False
        self.m.fit(fitter="mpfit", method="ls")
        for result, expected in zip(
                self.m[0].coefficients.value,
                (0.99165596295068958, 0.16628257462820528)):
            nt.assert_almost_equal(result, expected, places=5)

    def test_fit_fmin_unbinned(self):
        self.m.spectrum.metadata.Signal.binned = False
        self.m.fit(
            fitter="fmin",
            method="ls",
        )
        for result, expected in zip(
                self.m[0].coefficients.value,
                (0.99136169230026261, 0.18483060534056939)):
            nt.assert_almost_equal(result, expected, places=5)

    def test_chisq(self):
        self.m.spectrum.metadata.Signal.binned = True
        self.m.fit(fitter="leastsq", method="ls")
        nt.assert_almost_equal(self.m.chisq.data, 3029.16949561)

    def test_red_chisq(self):
        self.m.fit(fitter="leastsq", method="ls")
        nt.assert_almost_equal(self.m.red_chisq.data, 3.37700055)


class TestModelScalarVariance:

    def setUp(self):
        s = hs.signals.SpectrumSimulation(np.ones(100))
        m = s.create_model()
        m.append(hs.model.components.Offset())
        self.s = s
        self.m = m

    def test_std1_chisq(self):
        std = 1
        np.random.seed(1)
        self.s.add_gaussian_noise(std)
        self.s.metadata.set_item("Signal.Noise_properties.variance", std ** 2)
        self.m.fit(fitter="leastsq", method="ls")
        nt.assert_almost_equals(self.m.chisq.data, 78.35015229)

    def test_std10_chisq(self):
        std = 10
        np.random.seed(1)
        self.s.add_gaussian_noise(std)
        self.s.metadata.set_item("Signal.Noise_properties.variance", std ** 2)
        self.m.fit(fitter="leastsq", method="ls")
        nt.assert_almost_equals(self.m.chisq.data, 78.35015229)

    def test_std1_red_chisq(self):
        std = 1
        np.random.seed(1)
        self.s.add_gaussian_noise(std)
        self.s.metadata.set_item("Signal.Noise_properties.variance", std ** 2)
        self.m.fit(fitter="leastsq", method="ls")
        nt.assert_almost_equals(self.m.red_chisq.data, 0.79949135)

    def test_std10_red_chisq(self):
        std = 10
        np.random.seed(1)
        self.s.add_gaussian_noise(std)
        self.s.metadata.set_item("Signal.Noise_properties.variance", std ** 2)
        self.m.fit(fitter="leastsq", method="ls")
        nt.assert_almost_equals(self.m.red_chisq.data, 0.79949135)

    def test_std1_red_chisq_in_range(self):
        std = 1
        self.m.set_signal_range(10, 50)
        np.random.seed(1)
        self.s.add_gaussian_noise(std)
        self.s.metadata.set_item("Signal.Noise_properties.variance", std ** 2)
        self.m.fit(fitter="leastsq", method="ls")
        nt.assert_almost_equals(self.m.red_chisq.data, 0.86206965)


class TestModelSignalVariance:

    def setUp(self):
        variance = hs.signals.SpectrumSimulation(
            np.arange(
                100, 300).reshape(
                (2, 100)))
        s = variance.deepcopy()
        np.random.seed(1)
        std = 10
        s.add_gaussian_noise(std)
        s.add_poissonian_noise()
        s.metadata.set_item("Signal.Noise_properties.variance",
                            variance + std ** 2)
        m = s.create_model()
        m.append(hs.model.components.Polynomial(order=1))
        self.s = s
        self.m = m

    def test_std1_red_chisq(self):
        self.m.multifit(fitter="leastsq", method="ls", show_progressbar=None)
        nt.assert_almost_equals(self.m.red_chisq.data[0],
                                0.79693355673230915)
        nt.assert_almost_equals(self.m.red_chisq.data[1],
                                0.91453032901427167)


class TestMultifit:

    def setUp(self):
        s = hs.signals.Spectrum(np.zeros((2, 200)))
        s.axes_manager[-1].offset = 1
        s.data[:] = 2 * s.axes_manager[-1].axis ** (-3)
        m = s.create_model()
        m.append(hs.model.components.PowerLaw())
        m[0].A.value = 2
        m[0].r.value = 2
        m.store_current_values()
        m.axes_manager.indices = (1,)
        m[0].r.value = 100
        m[0].A.value = 2
        m.store_current_values()
        m[0].A.free = False
        self.m = m
        m.axes_manager.indices = (0,)
        m[0].A.value = 100

    def test_fetch_only_fixed_false(self):
        self.m.multifit(fetch_only_fixed=False, show_progressbar=None)
        np.testing.assert_array_almost_equal(self.m[0].r.map['values'],
                                             [3., 100.])
        np.testing.assert_array_almost_equal(self.m[0].A.map['values'],
                                             [2., 2.])

    def test_fetch_only_fixed_true(self):
        self.m.multifit(fetch_only_fixed=True, show_progressbar=None)
        np.testing.assert_array_almost_equal(self.m[0].r.map['values'],
                                             [3., 3.])
        np.testing.assert_array_almost_equal(self.m[0].A.map['values'],
                                             [2., 2.])


class TestStoreCurrentValues:

    def setUp(self):
        self.m = hs.signals.Spectrum(np.arange(10)).create_model()
        self.o = hs.model.components.Offset()
        self.m.append(self.o)

    def test_active(self):
        self.o.offset.value = 2
        self.o.offset.std = 3
        self.m.store_current_values()
        nt.assert_equal(self.o.offset.map["values"][0], 2)
        nt.assert_equal(self.o.offset.map["is_set"][0], True)

    def test_not_active(self):
        self.o.active = False
        self.o.offset.value = 2
        self.o.offset.std = 3
        self.m.store_current_values()
        nt.assert_not_equal(self.o.offset.map["values"][0], 2)


class TestSetCurrentValuesTo:

    def setUp(self):
        self.m = hs.signals.Spectrum(
            np.arange(10).reshape(2, 5)).create_model()
        self.comps = [
            hs.model.components.Offset(),
            hs.model.components.Offset()]
        self.m.extend(self.comps)

    def test_set_all(self):
        for c in self.comps:
            c.offset.value = 2
        self.m.assign_current_values_to_all()
        nt.assert_true((self.comps[0].offset.map["values"] == 2).all())
        nt.assert_true((self.comps[1].offset.map["values"] == 2).all())

    def test_set_1(self):
        self.comps[1].offset.value = 2
        self.m.assign_current_values_to_all([self.comps[1]])
        nt.assert_true((self.comps[0].offset.map["values"] != 2).all())
        nt.assert_true((self.comps[1].offset.map["values"] == 2).all())


class TestAsSignal:

    def setUp(self):
        self.m = hs.signals.Spectrum(
            np.arange(10).reshape(2, 5)).create_model()
        self.comps = [
            hs.model.components.Offset(),
            hs.model.components.Offset()]
        self.m.extend(self.comps)
        for c in self.comps:
            c.offset.value = 2
        self.m.assign_current_values_to_all()

    def test_all_components_simple(self):
        s = self.m.as_signal(show_progressbar=None)
        nt.assert_true(np.all(s.data == 4.))

    def test_one_component_simple(self):
        s = self.m.as_signal(component_list=[0], show_progressbar=None)
        nt.assert_true(np.all(s.data == 2.))
        nt.assert_true(self.m[1].active)

    def test_all_components_multidim(self):
        self.m[0].active_is_multidimensional = True

        s = self.m.as_signal(show_progressbar=None)
        nt.assert_true(np.all(s.data == 4.))

        self.m[0]._active_array[0] = False
        s = self.m.as_signal(show_progressbar=None)
        np.testing.assert_array_equal(
            s.data, np.array([np.ones(5) * 2, np.ones(5) * 4]))
        nt.assert_true(self.m[0].active_is_multidimensional)

    def test_one_component_multidim(self):
        self.m[0].active_is_multidimensional = True

        s = self.m.as_signal(component_list=[0], show_progressbar=None)
        nt.assert_true(np.all(s.data == 2.))
        nt.assert_true(self.m[1].active)
        nt.assert_false(self.m[1].active_is_multidimensional)

        s = self.m.as_signal(component_list=[1], show_progressbar=None)
        np.testing.assert_equal(s.data, 2.)
        nt.assert_true(self.m[0].active_is_multidimensional)

        self.m[0]._active_array[0] = False
        s = self.m.as_signal(component_list=[1], show_progressbar=None)
        nt.assert_true(np.all(s.data == 2.))

        s = self.m.as_signal(component_list=[0], show_progressbar=None)
        nt.assert_true(
            np.all(s.data == np.array([np.zeros(5), np.ones(5) * 2])))


class TestCreateModel:

    def setUp(self):
        self.s = hs.signals.Spectrum(np.asarray([0, ]))
        self.im = hs.signals.Image(np.ones([1, 1, ]))

    def test_create_model(self):
        from hyperspy.models.model1D import Model1D
        from hyperspy.models.model2D import Model2D
        nt.assert_is_instance(
            self.s.create_model(), Model1D)
        nt.assert_is_instance(
            self.im.create_model(), Model2D)


class TestAdjustPosition:

    def setUp(self):
        self.s = hs.signals.Spectrum(np.random.rand(10, 10, 20))
        self.m = self.s.create_model()

    def test_enable_adjust_position(self):
        self.m.append(hs.model.components.Gaussian())
        self.m.enable_adjust_position()
        nt.assert_equal(len(self.m._position_widgets), 1)
        # Check that both line and label was added
        nt.assert_equal(len(self.m._position_widgets.values()[0]), 2)

    def test_disable_adjust_position(self):
        self.m.append(hs.model.components.Gaussian())
        self.m.enable_adjust_position()
        self.m.disable_adjust_position()
        nt.assert_equal(len(self.m._position_widgets), 0)

    def test_enable_all(self):
        self.m.append(hs.model.components.Gaussian())
        self.m.enable_adjust_position()
        self.m.append(hs.model.components.Gaussian())
        nt.assert_equal(len(self.m._position_widgets), 2)

    def test_enable_all_zero_start(self):
        self.m.enable_adjust_position()
        self.m.append(hs.model.components.Gaussian())
        nt.assert_equal(len(self.m._position_widgets), 1)

    def test_manual_close(self):
        self.m.append(hs.model.components.Gaussian())
        self.m.append(hs.model.components.Gaussian())
        self.m.enable_adjust_position()
        self.m._position_widgets.values()[0][0].close()
        nt.assert_equal(len(self.m._position_widgets), 2)
        nt.assert_equal(len(self.m._position_widgets.values()[0]), 1)
        self.m._position_widgets.values()[0][0].close()
        nt.assert_equal(len(self.m._position_widgets), 1)
        nt.assert_equal(len(self.m._position_widgets.values()[0]), 2)
        self.m.disable_adjust_position()
        nt.assert_equal(len(self.m._position_widgets), 0)
