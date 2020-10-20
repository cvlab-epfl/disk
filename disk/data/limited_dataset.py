import warnings, random

from torch.utils.data import ConcatDataset

from disk import DataError

class LimitableDataset:
    def __init__(self, inner, warn=False):
        self._indexes   = list(range(len(inner)))
        self._yieldable = self._indexes
        self._inner     = inner

        self.warn       = warn

    def limit(self, n, shuffle=False):
        yieldable = self._indexes[:]

        if shuffle:
            random.shuffle(yieldable)

        if n is not None and len(yieldable) < n:
            msg = (f'Trying to limit a dataset to {n} items, '
                   f'only has {len(yieldable)} in total')

            if self.warn:
                warnings.warn(msg)
            else:
                raise DataError(msg)

        self._yieldable = yieldable[:n]

    def __len__(self):
        return len(self._yieldable)

    def __getitem__(self, idx):
        return self._inner[self._yieldable[idx]]

class LimitedConcatDataset(ConcatDataset):
    def __init__(self, datasets, limit=None, shuffle=False):
        self.limit = limit

        limitables = [LimitableDataset(ds) for ds in datasets]

        for ds in limitables:
            ds.limit(limit, shuffle=shuffle)

        super(LimitedConcatDataset, self).__init__(limitables)

    def shuffle(self):
        for ds in self.datasets:
            ds.limit(self.limit, shuffle=True)
