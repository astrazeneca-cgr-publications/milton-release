"""Simple utitilties for global variables.
"""


class Proxy:
    """A proxy to an object that forwards all method calls while
    allowing for object substitution. To be used with a context
    manager as a more civilized global variable.
    """
    
    def __init__(self, obj):
        self._object = obj
        
    def set_global(self, new_obj):
        old = self._object
        self._object = new_obj
        return old
        
    def __getattr__(self, name):
        return getattr(self._object, name)
    
    def __call__(self, *args, **kwargs):
        return self._object(*args, **kwargs)
    
    def __str__(self):
        return str(self._object)
    
    def __repr__(self):
        return self._object.__repr__()
    
    def __dir__(self):
        return self._object.__dir__()
    

# Global UkbDataSource proxy
DS = Proxy(None)

# Global UkbDataStore proxy
DST = Proxy(None)

# Global UkbDataDictionary proxy
DD = Proxy(None)

# Global Dask Client proxy
DASK = Proxy(None)

# Global patient selector
SEL = Proxy(None)
