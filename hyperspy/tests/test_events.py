import nose.tools as nt
import hyperspy.events as he
import sys
from contextlib import contextmanager


@contextmanager
def nostderr():
    savestderr = sys.stderr
    savestdout = sys.stdout

    class Devnull(object):
        def write(self, _): pass

        def flush(self): pass

    sys.stderr = Devnull()
    sys.stdout = Devnull()
    try:
        yield
    finally:
        sys.stderr = savestderr
        sys.stdout = savestdout


class EventsBase():
    def on_trigger(self, *args, **kwargs):
        self.triggered = True
        self.trigger_args = args

    def on_trigger2(self, *args, **kwargs):
        self.triggered2 = True

    def trigger_check(self, trigger, should_trigger, *args):
        self.triggered = False
        trigger(*args)
        nt.assert_equal(self.triggered, should_trigger)

    def trigger_check2(self, trigger, should_trigger, *args):
        self.triggered2 = False
        trigger(*args)
        nt.assert_equal(self.triggered2, should_trigger)

    def trigger_check_args(self, trigger, should_trigger, expected_args,
                           *args):
        self.trigger_args = None
        self.trigger_check(trigger, should_trigger, *args)
        nt.assert_equal(self.trigger_args, expected_args)


class TestEventsSuppression(EventsBase):

    def setUp(self):
        self.events = he.Events()

        self.events.a = he.Event()
        self.events.b = he.Event()
        self.events.c = he.Event()

        self.events.a.connect(self.on_trigger)
        self.events.a.connect(self.on_trigger2)
        self.events.b.connect(self.on_trigger)
        self.events.c.connect(self.on_trigger)

    def test_simple_suppression(self):
        with self.events.a.suppress():
            self.trigger_check(self.events.a.trigger, False)
            self.trigger_check(self.events.b.trigger, True)

        with self.events.suppress():
            self.trigger_check(self.events.a.trigger, False)
            self.trigger_check(self.events.b.trigger, False)
            self.trigger_check(self.events.c.trigger, False)

        self.trigger_check(self.events.a.trigger, True)
        self.trigger_check(self.events.b.trigger, True)
        self.trigger_check(self.events.c.trigger, True)

    def test_suppression_restore(self):
        with self.events.a.suppress():
            with self.events.suppress():
                self.trigger_check(self.events.a.trigger, False)
                self.trigger_check(self.events.b.trigger, False)
                self.trigger_check(self.events.c.trigger, False)

            self.trigger_check(self.events.a.trigger, False)
            self.trigger_check(self.events.b.trigger, True)
            self.trigger_check(self.events.c.trigger, True)

    def test_suppresion_nesting(self):
        with self.events.a.suppress():
            with self.events.suppress():
                self.events.c._suppress = False
                self.trigger_check(self.events.a.trigger, False)
                self.trigger_check(self.events.b.trigger, False)
                self.trigger_check(self.events.c.trigger, True)

                with self.events.suppress():
                    self.trigger_check(self.events.a.trigger, False)
                    self.trigger_check(self.events.b.trigger, False)
                    self.trigger_check(self.events.c.trigger, False)

                self.trigger_check(self.events.a.trigger, False)
                self.trigger_check(self.events.b.trigger, False)
                self.trigger_check(self.events.c.trigger, True)

            self.trigger_check(self.events.a.trigger, False)
            self.trigger_check(self.events.b.trigger, True)
            self.trigger_check(self.events.c.trigger, True)

    def test_suppression_single(self):
        with self.events.b.suppress():
            with self.events.a.suppress_callback(self.on_trigger):
                self.trigger_check(self.events.a.trigger, False)
                self.trigger_check2(self.events.a.trigger, True)
                self.trigger_check(self.events.b.trigger, False)
                self.trigger_check(self.events.c.trigger, True)

            self.trigger_check(self.events.a.trigger, True)
            self.trigger_check2(self.events.a.trigger, True)
            self.trigger_check(self.events.b.trigger, False)
            self.trigger_check(self.events.c.trigger, True)

        # Reverse order:
        with self.events.a.suppress_callback(self.on_trigger):
            with self.events.b.suppress():
                self.trigger_check(self.events.a.trigger, False)
                self.trigger_check2(self.events.a.trigger, True)
                self.trigger_check(self.events.b.trigger, False)
                self.trigger_check(self.events.c.trigger, True)

            self.trigger_check(self.events.a.trigger, False)
            self.trigger_check2(self.events.a.trigger, True)
            self.trigger_check(self.events.b.trigger, True)
            self.trigger_check(self.events.c.trigger, True)

    @nt.raises(ValueError)
    def test_exception_event(self):
        try:
            with self.events.a.suppress():
                self.trigger_check(self.events.a.trigger, False)
                self.trigger_check(self.events.b.trigger, True)
                self.trigger_check(self.events.c.trigger, True)
                raise ValueError()
        finally:
            self.trigger_check(self.events.a.trigger, True)
            self.trigger_check(self.events.b.trigger, True)
            self.trigger_check(self.events.c.trigger, True)

    @nt.raises(ValueError)
    def test_exception_events(self):
        try:
            with self.events.suppress():
                self.trigger_check(self.events.a.trigger, False)
                self.trigger_check(self.events.b.trigger, False)
                self.trigger_check(self.events.c.trigger, False)
                raise ValueError()
        finally:
            self.trigger_check(self.events.a.trigger, True)
            self.trigger_check(self.events.b.trigger, True)
            self.trigger_check(self.events.c.trigger, True)

    @nt.raises(ValueError)
    def test_exception_single(self):
        try:
            with self.events.a.suppress_callback(self.on_trigger):
                self.trigger_check(self.events.a.trigger, False)
                self.trigger_check2(self.events.a.trigger, True)
                self.trigger_check(self.events.b.trigger, True)
                self.trigger_check(self.events.c.trigger, True)
                raise ValueError()
        finally:
            self.trigger_check(self.events.a.trigger, True)
            self.trigger_check2(self.events.a.trigger, True)
            self.trigger_check(self.events.b.trigger, True)
            self.trigger_check(self.events.c.trigger, True)

    @nt.raises(ValueError)
    def test_exception_nested(self):
        try:
            with self.events.a.suppress_callback(self.on_trigger):
                try:
                    with self.events.a.suppress():
                        try:
                            with self.events.suppress():
                                self.trigger_check(self.events.a.trigger,
                                                   False)
                                self.trigger_check2(self.events.a.trigger,
                                                    False)
                                self.trigger_check(self.events.b.trigger,
                                                   False)
                                self.trigger_check(self.events.c.trigger,
                                                   False)
                                raise ValueError()
                        finally:
                            self.trigger_check(self.events.a.trigger, False)
                            self.trigger_check2(self.events.a.trigger, False)
                            self.trigger_check(self.events.b.trigger, True)
                            self.trigger_check(self.events.c.trigger, True)
                finally:
                    self.trigger_check(self.events.a.trigger, False)
                    self.trigger_check2(self.events.a.trigger, True)
                    self.trigger_check(self.events.b.trigger, True)
                    self.trigger_check(self.events.c.trigger, True)
        finally:
            self.trigger_check(self.events.a.trigger, True)
            self.trigger_check2(self.events.a.trigger, True)
            self.trigger_check(self.events.b.trigger, True)
            self.trigger_check(self.events.c.trigger, True)

    def test_suppress_wrong(self):
        with self.events.a.suppress_callback(f_a):
            self.trigger_check(self.events.a.trigger, True)
            self.trigger_check2(self.events.a.trigger, True)


def f_a(*args): pass
def f_b(*args): pass
def f_c(*args): pass
def f_d(a, b, c): pass


class TestEventsSignatures(EventsBase):

    def setUp(self):
        self.events = he.Events()
        self.events.a = he.Event()

    def test_basic_triggers(self):
        self.events.a.connect(lambda *args, **kwargs: 0)
        self.events.a.connect(lambda: 0, None)
        self.events.a.connect(lambda x: 0, 1)
        self.events.a.connect(lambda x, y: 0, 2)
        self.events.a.connect(lambda x, y=988:
                              nt.assert_equal(y, 988), 1)
        self.events.a.connect(lambda x, y=988:
                              nt.assert_not_equal(y, 988), 2)
        self.events.a.trigger(2, 5)

        nt.assert_raises(ValueError, self.events.a.trigger)
        nt.assert_raises(ValueError, self.events.a.trigger, 2)
        self.events.a.trigger(2, 5, 8)

    def test_connected(self):
        self.events.a.connect(f_a)
        self.events.a.connect(f_a, None)
        self.events.a.connect(f_b, 2)
        self.events.a.connect(f_c, 5)
        self.events.a.connect(f_d, 'auto')

        ref = {'all': set([f_a]), 0: set([f_a]), 1: set(), 2: set([f_b]),
               3: set([f_d]), 5: set([f_c]),
               None: set([f_a, f_b, f_c, f_d])}
        for k, v in ref.iteritems():
            con = self.events.a.connected(k)
            nt.assert_equal(con, v)

        con = self.events.a.connected()
        nt.assert_equal(con, ref[None])

    @nt.raises(TypeError)
    def test_type(self):
        self.events.a.connect('f_a')

def _trig_d(obj):
    obj.d = 'Ohmy'


def _trig_a(obj):
    obj.a = 1.57


class TestTraitsEvents(EventsBase):

    def setUp(self):
        import traits.api as t

        class dummy_simple(he.HasEventsTraits):
            d = t.String('Tester')

        class dummy(he.HasEventsTraits):
            a = t.CFloat(5.)
            b = t.CInt()
            c = t.Instance(dummy_simple)

            def __init__(self):
                super(dummy, self).__init__()
                self.c = dummy_simple()

        self.obj = dummy_simple()
        self.adv = dummy()

    def test_simple_0(self):
        self.obj.events.d_changed.connect(self.on_trigger, 0)
        self.trigger_check_args(_trig_d, True, tuple(), self.obj)
        self.obj.events.d_changed.disconnect(self.on_trigger)
        self.trigger_check(_trig_d, False, self.obj)

    def test_simple_1(self):
        self.obj.events.d_changed.connect(self.on_trigger, 1)
        self.trigger_check_args(_trig_d, True, ('Ohmy',), self.obj)
        self.obj.events.d_changed.disconnect(self.on_trigger)
        self.trigger_check(_trig_d, False, self.obj)

    def test_simple_2(self):
        self.obj.events.d_changed.connect(self.on_trigger, 2)
        self.trigger_check_args(_trig_d, True, ('d', 'Ohmy'), self.obj)
        self.obj.events.d_changed.disconnect(self.on_trigger)
        self.trigger_check(_trig_d, False, self.obj)

    def test_simple_3(self):
        self.obj.events.d_changed.connect(self.on_trigger, 3)
        self.trigger_check_args(_trig_d, True, (self.obj, 'd', 'Ohmy'),
                                self.obj)
        self.obj.events.d_changed.disconnect(self.on_trigger)
        self.trigger_check(_trig_d, False, self.obj)

    def test_simple_4(self):
        self.obj.events.d_changed.connect(self.on_trigger, 4)
        self.trigger_check_args(_trig_d, True,
                                (self.obj, 'd', 'Tester', 'Ohmy'), self.obj)
        self.obj.events.d_changed.disconnect(self.on_trigger)
        self.trigger_check(_trig_d, False, self.obj)

    def test_simple_5(self):
        self.obj.events.d_changed.connect(self.on_trigger, 5)
        # Should fail, since we don't support 5 args
        with nostderr():
            self.trigger_check(_trig_d, False, self.obj)
        self.obj.events.d_changed.disconnect(self.on_trigger)
        self.trigger_check(_trig_d, False, self.obj)

    def test_simple_all(self):
        self.obj.events.d_changed.connect(self.on_trigger, 'all')
        self.trigger_check_args(_trig_d, True,
                                (self.obj, 'd', 'Tester', 'Ohmy'), self.obj)
        self.obj.events.d_changed.disconnect(self.on_trigger)
        self.trigger_check(_trig_d, False, self.obj)

    def test_adv(self):
        self.adv.events.a_changed.connect(self.on_trigger, 1)
        self.trigger_check_args(_trig_a, True, (1.57,), self.adv)
        self.adv.events.a_changed.disconnect(self.on_trigger)
        self.trigger_check(_trig_a, False, self.adv)

    def test_adv_sub(self):
        self.adv.c.events.d_changed.connect(self.on_trigger, 1)
        self.trigger_check_args(_trig_d, True, ('Ohmy',), self.adv.c)
        self.adv.c.events.d_changed.disconnect(self.on_trigger)
        self.trigger_check(_trig_d, False, self.adv.c)
