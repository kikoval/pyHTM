def GetSomeVars(self, names):
    """Grab selected attributes from an object.
    Used during pickling.

    Params:
        names: A sequence of strings, each an attribute name strings

    Returns:
        a dictionary of member variable values
    """
    return {i: getattr(self, i) for i in names}


def CallReader(self, readFunc, parents, state):
    """Call the function specified by name string readFunc on self
    with arguments parents and state: i.e.
    self.readFunc(parents, state)

    Params:
        self:     Object to call readFunc callable on
        readFunc: String name of a callable bound to self
        parents:  Arbitrary argument to be passed to the callable
        state:    Arbitrary argument to be passed to the callable
    """
    getattr(self, readFunc)(parents, state)


def UpdateMembers(self, state):
    """Update attributes of self named in the dictionary keys in state
    with the values in state.

    Params:
        self: Object whose attributes will be updated
        state: Dictionary of attribute names and values
    """
    for i in state.keys():
        setattr(self, i, state[i])
