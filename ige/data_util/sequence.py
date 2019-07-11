from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections


def _assert_sequence(value, name):
    for attr in ('__len__', '__getitem__'):
        if not hasattr(value, '__len__'):
            raise ValueError('item %s missing attr %s' % (name, attr))


def _check_items(items):
    n = None
    for k, v in items:
        _assert_sequence(v, k)
        if n is None:
            n = len(v)
            base_k = k
        elif len(v) != n:
            raise ValueError(
                'item %s has different length to %s' % (k, base_k))
    return n 


class CompoundSequence(collections.Sequence):
    def __init__(self, items):
        self._len = _check_items(items())
        if self._len is None:
            raise ValueError('must provide at least one kwargs')

    def __len__(self):
        return self._len


class DictSequence(CompoundSequence):
    """
    Convert a dict of sequences to a sequence of dicts.

    Example usage:
    ```python
    x = [0, 1, 2, 3]
    y = [10, 11, 12, 13]
    dict_seq = DictSequence(x=x, y=y)
    dict_seq[2] == dict(x=2, y=12)
    len(dict_seq) == 4
    ```
    """
    def __init__(self, **kwargs):
        super(DictSequence, self).__init__(kwargs.items())
        self._sequences = kwargs
    
    def __getitem__(self, index):
        return {k: v[index] for k, v in self._sequences.items()}


class ZippedSequence(CompoundSequence):
    """
    Sequence interface for zipped sequences.

    Example usage:
    ```python
    x = [0, 1, 2, 3]
    y = [10, 11, 12, 13]
    zipped = ZippedSequence(x, y)
    zipped[2] == (2, 12)
    len(zipped) == 4
    ```
    """
    def __init__(self, *args):
        super(ZippedSequence, self).__init__(enumerate(args))
        self._sequences = args
    
    def __getitem__(self, index):
        return tuple(sequence[index] for sequence in self._sequences)


class MappedSequence(collections.Sequence):
    """
    Lazily mapped sequence.

    Example usage:
    ```python
    x = [0, 2, 5]
    m = MappedSequence(x, lambda x: x**2)
    m[1] == 4
    len(m) == 3
    list(m) == [0, 4, 25]
    ```
    """
    def __init__(self, sequence, map_fn):
        _assert_sequence(sequence, 'sequence')
        self._sequence = sequence
        self._map_fn = map_fn
    
    def __len__(self):
        return len(self._sequence)
    
    def __getitem__(self, index):
        return self._map_fn(self._sequence[index])


def zipped(*args, **kwargs):
    """
    Convenience interface to `DictSequence`/`ZippedSequence`.

    One of `args` or `kwargs` must be empty.
    """
    if len(args) == 0:
        return DictSequence(**kwargs)
    elif len(kwargs) == 0:
        return ZippedSequence(*args)
    else:
        raise ValueError('Either args or kwargs must be empty')


mapped = MappedSequence