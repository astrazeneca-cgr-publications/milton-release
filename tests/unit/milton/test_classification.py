import pytest
import numpy as np
from milton.classification import (
    array_splits,
    balanced_partitions)


def test_array_splits():
    def assert_equal(splits, expected):
        splits = list(splits)
        assert len(splits) == len(expected)
        for a, b in zip(splits, expected):
            assert np.all(np.equal(a, b))
    
    ix = np.arange(10)
    assert_equal(array_splits(ix, 5), [np.arange(5), np.arange(5, 10)])
    assert_equal(array_splits(ix, 4), [np.arange(5), np.arange(5, 10)])
    assert_equal(array_splits(ix, 10), [ix])
    
    ix = np.arange(11)
    assert_equal(array_splits(ix, 3), [np.arange(3), np.arange(3, 6), 
                                       np.arange(6, 8), np.arange(8, 11)])
    
    ix = np.arange(100)
    assert_equal(array_splits(ix, 38), [np.arange(33), np.arange(33, 67), 
                                        np.arange(67, 100)])
    assert_equal(array_splits(ix, 42), [np.arange(50), np.arange(50, 100)])


@pytest.mark.parametrize('case_n, ctl_n, expected_len', [
    (39, 100, 3),
    (100, 39, 3),
    (41, 100, 2),
    (100, 41, 2)
])
def test_balanced_partitions(case_n, ctl_n, expected_len):
    y = np.zeros(case_n + ctl_n)
    y[:case_n] = 1
    
    left = set()
    right = set()
    partitions = list(balanced_partitions(y))
    
    assert len(partitions) == expected_len
    
    for a, b in partitions:
        left.add(tuple(a))
        right.add(tuple(b))
        
    if case_n < ctl_n:
        assert len(left) == 1
        assert len(partitions) == len(right)
    else:
        assert len(right) == 1
        assert len(partitions) == len(left)
