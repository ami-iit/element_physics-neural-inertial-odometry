class dotdict(dict):
    r"""
    A simple subclass of Python's built-in dict that allows attribute-style access
    to dict keys, e.g., obj.key instead of obj['key'].
    """
    def __setattr__(self, name, value):
        """Instead of storing attributes in self.__dict__, stors them as dict keys."""
        self[name] = value 
    def __getattr__(self, name):
        """
        If an attribute isn't found in the normal way, try to retrieve it from the dict,
        if the key does not exist, raise an error.
        """
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)
    def __repr__(self):
        """Return a string representation of the dict."""
        return super().__repr__()