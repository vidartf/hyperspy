import sys
import inspect
from traits.api import HasTraits, MetaHasTraits


class EventsSuppressionContext(object):

    """
    Context manager for event suppression. When passed an Events class,
    it will suppress all the events in that container when activated by
    using it in a 'with' statement. The previous suppression state will be
    restored when the 'with' block completes, allowing for nested suppression.
    """

    def __init__(self, events):
        self.events = events
        self.old = {}

    def __enter__(self):
        self.old = {}
        try:
            for e in self.events.__dict__.itervalues():
                self.old[e] = e._suppress
                e._suppress = True
        except:
            self.__exit__(*sys.exc_info())
            raise
        return self

    def __exit__(self, type, value, tb):
        for e, oldval in self.old.iteritems():
            e._suppress = oldval
        # Never suppress exceptions


class EventSuppressionContext(object):

    """
    Context manager for event suppression. When passed an Event class,
    it will suppress the event when activated by using it in a 'with'
    statement. The previous suppression state will be restored when the 'with'
    block completes, allowing for nested suppression.
    """

    def __init__(self, event):
        self.event = event
        self.old = None

    def __enter__(self):
        self.old = None
        try:
            self.old = self.event._suppress
            self.event._suppress = True
        except:
            self.__exit__(*sys.exc_info())
            raise
        return self

    def __exit__(self, type, value, tb):
        if self.old is not None:
            self.event._suppress = self.old
        # Never suppress exceptions


class CallbackSuppressionContext(object):

    """
    Context manager for suppression of a single callback on an Event. Useful
    e.g. to prevent infinite recursion if two objects are connected in a loop.
    """

    def __init__(self, callback, event, nargs):
        self.event = event
        self.callback = callback
        self.nargs = nargs

    def __enter__(self):
        if self.callback is not None:
            self.event.disconnect(self.callback)

    def __exit__(self, type, value, tb):
        if self.callback is not None:
            self.event.connect(self.callback, self.nargs)


class Events(object):

    """
    Events container.

    All available events are attributes of this class.

    """

    def suppress(self):
        """
        Use this function with a 'with' statement to temporarily suppress
        all events in the container. When the 'with' lock completes, the old
        suppression values will be restored.

        Example usage pattern:
        with obj.events.suppress:
            obj.val_a = a
            obj.val_b = b
        obj.events.values_changed.trigger()
        """
        return EventsSuppressionContext(self)


class Event(object):

    def __init__(self):
        self._connected = {}
        self._suppress = False

    def suppress(self):
        """
        Use this function with a 'with' statement to temporarily suppress
        all events in the container. When the 'with' lock completes, the old
        suppression values will be restored.

        Example usage pattern:
        with obj.events.myevent.suppress():
            obj.val_a = a
            obj.val_b = b
        obj.events.myevent.trigger()
        """
        return EventSuppressionContext(self)

    def suppress_callback(self, function):
        """
        Use this function with a 'with' statement to temporarily suppress
        a single callback from being called. All other connected callbacks
        will trigger. When the 'with' lock completes, the old suppression value
        will be restored.
        """
        nargs = None
        found = False
        for nargs, c in self._connected.iteritems():
            for f in c:
                if f == function:
                    found = True
                    break
        if not found:
            function = None
        return CallbackSuppressionContext(function, self, nargs)

    def connected(self, nargs=None):
        """
        Connected functions. The default behavior is to include all
        functions, but by using the 'nargs' argument, it can be filtered by
        function signature.
        """
        if nargs is None:
            ret = set()
            ret.update(*self._connected.values())
            return ret
        else:
            if nargs in self._connected:
                return self._connected[nargs]
            else:
                return set()

    def connect(self, function, nargs='all'):
        """
        Connects a function to the event.
        Arguments:
        ----------
        function : callable
            The function to call when the event triggers.
        nargs : int, 'all' (default), or 'auto'
            The number of arguments to supply to the function. If 'all', it
            will be called with all arguments passed to trigger(). If 'auto'
            inspect.getargspec() will be used to determine the number of
            arguments the function accepts (arguments with default values will
            be included in the count).
        """
        if not callable(function):
            raise TypeError("Only callables can be registered")
        if nargs == 'auto':
            spec = inspect.getargspec(function)[0]
            if spec is None:
                nargs = 0
            else:
                nargs = len(spec)
        elif nargs is None:
            nargs = 0
        if nargs not in self._connected:
            self._connected[nargs] = set()
        self._connected[nargs].add(function)

    def disconnect(self, function):
        """
        Disconnects a function from the event. The passed function will be
        disconnected irregardless of which 'nargs' argument was passed to
        connect().
        """
        for c in self._connected.itervalues():
            if function in c:
                c.remove(function)

    @staticmethod
    def _trigger_nargs(f, args, nargs):
        """
        Basic trigger resolution.
        """
        return f(*args[0:nargs])

    def trigger(self, *args, **kwargs):
        if not self._suppress:
            # Loop on copy to deal with callbacks which change connections
            for nargs, c in self._connected.copy().iteritems():
                if nargs is 'all':
                    for f in c:
                        f(*args, **kwargs)
                else:
                    if len(args) < nargs:
                        raise ValueError(
                            ("Tried to call %s which require %d args " +
                             "with only %d.") % (str(c), nargs, len(args)))
                    for f in c.copy():
                        self._trigger_nargs(f, args, nargs)

    def __deepcopy__(self, memo):
        dc = type(self)()
        memo[id(self)] = dc
        return dc


class TraitEvent(Event):

    def trigger(self, obj, name, old, new):
        super(TraitEvent, self).trigger(obj, name, old, new)

    @staticmethod
    def _trigger_nargs(f, args, nargs):
        """
        Emulates traits resolution:
            handler()
            handler(new)
            handler(name, new)
            handler(object, name, new)
            handler(object, name, old, new)
        """
        if nargs == 0:
            return f()
        elif nargs == 1:
            return f(args[3])
        elif nargs == 2:
            return f(args[1], args[3])
        elif nargs == 3:
            return f(args[0], args[1], args[3])
        elif nargs == 4:
            return f(*args)


traits_filter = ['trait_added', 'trait_modified', 'events']


def _wrap_trait(collection, obj, name):
    e = TraitEvent()
    setattr(collection, name + "_changed", e)
    obj.on_trait_change(e.trigger, name)


def _init_wrap(self, *args, **kwargs):
    if self._old_init is not None:
        self._old_init(*args, **kwargs)

    # Setup events container if missing
    if not hasattr(self, 'events'):
        self.events = Events()

    # Wrap traits
    if isinstance(self, HasTraits):
        for t in self.traits().iterkeys():
            if t in traits_filter:
                continue
            _wrap_trait(self.events, self, t)


class MetaHasEventsTraits(MetaHasTraits):

    def __new__(cls, name, bases, attrs):
        attrs['_old_init'] = attrs.pop('__init__', None)
        attrs['__init__'] = _init_wrap
        return super(MetaHasEventsTraits, cls).__new__(cls, name, bases, attrs)


class HasEventsTraits(HasTraits):
    __metaclass__ = MetaHasEventsTraits
