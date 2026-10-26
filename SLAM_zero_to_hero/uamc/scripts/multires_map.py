"""Multi-resolution, view-adaptive voxel averaging of a coloured point stream.

Points arrive one batch (one "view") at a time. Each batch is reduced into the finest cells
(min_leaf): sum of xyz, sum of BGR and point count. For every level l (cell size
min_leaf * 2**l, up to max_leaf) the number of distinct batches that hit each cell is counted:
that is the cell's view count.

build() walks the octree from max_leaf down. A child cell seen by at least split_views
batches is "strong"; a cell with any strong child is split: strong children go one level
down, its weak children are merged into one averaged point. Unsplit cells, and strong cells
at min_leaf, become one averaged point each. Places many views agree on end up dense and
averaged at fine cells; rarely seen places stay as one point per coarse cell.

Accumulators are sorted key arrays merged when the pending part reaches half their size, so
the total merge work stays a small multiple of one final sort.
"""
import numpy as np

B = 1 << 20                      # per-axis key offset; 21 bits per axis
M21 = (1 << 21) - 1


def pack(k):
    return (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]


def unpack(key):
    return np.stack([(key >> 42) & M21, (key >> 21) & M21, key & M21], 1)


def parent(key, n=1):
    return pack(unpack(key) >> n)


class _Acc:
    """Sorted unique keys with summed value columns; adds are buffered and merged in bulk."""

    def __init__(self, ncol):
        self.k = np.zeros(0, np.int64)
        self.v = np.zeros((0, ncol))
        self.pk, self.pv, self.npend = [], [], 0

    def add(self, k, v):
        self.pk.append(k); self.pv.append(v); self.npend += len(k)
        if self.npend >= max(1 << 20, len(self.k) // 2):
            self.merge()

    def merge(self):
        if not self.pk:
            return
        k = np.concatenate([self.k] + self.pk)
        v = np.concatenate([self.v] + self.pv)
        self.pk, self.pv, self.npend = [], [], 0
        u, inv = np.unique(k, return_inverse=True)
        out = np.empty((len(u), v.shape[1]))
        for c in range(v.shape[1]):
            out[:, c] = np.bincount(inv, weights=v[:, c], minlength=len(u))
        self.k, self.v = u, out


class MultiResMap:
    def __init__(self, min_leaf=0.005, max_leaf=0.16, split_views=3):
        self.min_leaf = min_leaf
        self.L = max(int(round(np.log2(max_leaf / min_leaf))), 0)
        self.split_views = split_views
        self.sums = _Acc(7)                                 # finest: sum xyz, sum bgr, n
        self.views = [_Acc(1) for _ in range(self.L + 1)]   # per level: distinct batches

    def add(self, xyz, bgr):
        """One batch (one view) of world points and their BGR colours."""
        if len(xyz) == 0:
            return
        k0 = pack(np.floor(xyz / self.min_leaf).astype(np.int64) + B)
        u, inv = np.unique(k0, return_inverse=True)
        v = np.empty((len(u), 7))
        for c in range(3):
            v[:, c] = np.bincount(inv, weights=xyz[:, c], minlength=len(u))
            v[:, 3 + c] = np.bincount(inv, weights=bgr[:, c], minlength=len(u))
        v[:, 6] = np.bincount(inv, minlength=len(u))
        self.sums.add(u, v)
        for l in range(self.L + 1):
            if l:
                u = np.unique(parent(u))
            self.views[l].add(u, np.ones((len(u), 1)))

    def n_cells(self):
        self.sums.merge()
        return len(self.sums.k)

    def build(self):
        """(xyz, bgr, level, n points) of the adaptive map; level 0 is min_leaf."""
        self.sums.merge()
        for a in self.views:
            a.merge()
        # sums per level, bottom-up from the finest cells
        keys, sums = [self.sums.k], [self.sums.v]
        for l in range(1, self.L + 1):
            u, inv = np.unique(parent(keys[-1]), return_inverse=True)
            s = np.empty((len(u), 7))
            for c in range(7):
                s[:, c] = np.bincount(inv, weights=sums[-1][:, c], minlength=len(u))
            keys.append(u); sums.append(s)
        views = []
        for l in range(self.L + 1):                         # align view counts to keys[l]
            j = np.searchsorted(self.views[l].k, keys[l])
            views.append(self.views[l].v[j, 0])

        out_s, out_l = [], []
        active = keys[self.L]                               # sorted
        for l in range(self.L, 0, -1):
            if len(active) == 0:
                break
            ck = keys[l - 1]
            pk = parent(ck)
            j = np.minimum(np.searchsorted(active, pk), len(active) - 1)
            inact = active[j] == pk
            ck, cs, cv, pidx = ck[inact], sums[l - 1][inact], views[l - 1][inact], j[inact]
            strong = cv >= self.split_views
            split = np.bincount(pidx, weights=strong, minlength=len(active)) > 0
            # unsplit active cells: one point each, from this level's sums
            ia = np.searchsorted(keys[l], active)
            out_s.append(sums[l][ia[~split]]); out_l.append(np.full((~split).sum(), l))
            # weak children of split cells: merged into one point per parent
            weak = ~strong & split[pidx]
            if weak.any():
                u, inv = np.unique(pidx[weak], return_inverse=True)
                ws = np.empty((len(u), 7))
                for c in range(7):
                    ws[:, c] = np.bincount(inv, weights=cs[weak][:, c], minlength=len(u))
                out_s.append(ws); out_l.append(np.full(len(u), l))
            active = ck[strong & split[pidx]]               # stays sorted (subset of sorted keys)
        ia = np.searchsorted(keys[0], active)
        out_s.append(sums[0][ia]); out_l.append(np.zeros(len(active), int))

        s = np.concatenate(out_s); lev = np.concatenate(out_l)
        n = s[:, 6:7]
        return s[:, 0:3] / n, s[:, 3:6] / n, lev, s[:, 6]
