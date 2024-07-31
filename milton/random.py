"""Global shared random state.
"""
import numpy as np
from contextlib import contextmanager
import hashlib
import pickle
import io


_RANDOM_STATE = np.random.default_rng(123)


def obj_to_int(obj):
    """MD5 digest from the pickled object as a 128-bit integer.
    """
    if not isinstance(obj, int):
        with io.BytesIO() as f:
            pickle.dump(obj, f)
            f.seek(0)
            digest = hashlib.md5(f.read()).digest()
            return int.from_bytes(digest, 'big')
    else:
        return obj
    

def new_rnd(seed_object):
    """Creates a new random state seeded with given object. Anything can be 
    passed for as long as it can be pickled.
    """
    seed = obj_to_int(seed_object)
    return np.random.default_rng(seed)


def randint():
    """Shortcut for producing a single random integer in the range [0, 2**32)
    using the current random generator.
    """
    return _RANDOM_STATE.integers(2**32, size=1)[0]


def RND() -> np.random.Generator:
    """The current random state object (Generator).
    """
    return _RANDOM_STATE


@contextmanager
def set_random_state(state):
    """Context manager that sets random state for all methods used by Milton
    to a specific seed. If the state is np.random.Generator, it is used directly.
    Otherwise, a new generator is created and seeded with the state.
    """
    global _RANDOM_STATE
    old_rnd = _RANDOM_STATE
    try:
        if isinstance(state, np.random.Generator):
            _RANDOM_STATE = state
        else:
            _RANDOM_STATE = new_rnd(state)
        yield _RANDOM_STATE
    finally:
        _RANDOM_STATE = old_rnd
