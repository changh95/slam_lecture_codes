"""Occupancy-grid quality metrics for a Cartographer ROS-map .pgm.

Classification reproduces exactly the numbers reported for the OLD 2D config
(occupied 37,715 = 5.58 %, free 32,415 = 4.79 %, unknown 71.1 % on the 859x787
grid): Cartographer writes 128 for UNKNOWN cells and otherwise a greyscale value
where LOW = occupied and HIGH = free. The old measurement used
    occupied := value < 100      free := value > 200      unknown := value == 128
(solved back out of the old .pgm histogram: cum(<=99) == 37715 and
 cum(>=201) == 32415 to the cell).

Also reports occupied horizontal run lengths in metres -- crisp walls give short
runs plus a few long straight ones; a smeared map gives fat slabs.
"""
import sys
import numpy as np


def readpgm(path):
    f = open(path, "rb")

    def tok():
        t = b""
        while True:
            c = f.read(1)
            if c == b"":
                raise EOFError
            if c == b"#":
                while f.read(1) not in (b"\n", b""):
                    pass
                continue
            if c.isspace():
                if t:
                    return t
                continue
            t += c

    magic = tok()
    assert magic == b"P5", magic
    w = int(tok()); h = int(tok()); int(tok())
    a = np.frombuffer(f.read(w * h), dtype=np.uint8).reshape(h, w)
    return a


def runs(mask):
    out = []
    for row in mask:
        idx = np.flatnonzero(np.diff(np.concatenate(([0], row.view(np.int8), [0]))))
        for s, e in zip(idx[0::2], idx[1::2]):
            out.append(e - s)
    return np.array(out)


path = sys.argv[1]
res = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
a = readpgm(path)
h, w = a.shape
N = a.size
occ = a < 100
free = a > 200
unk = a == 128
mid = ~(occ | free | unk)
print("file: %s" % path)
print("grid: %dx%d px at %.3f m/px  (%.1f x %.1f m)  %d cells"
      % (w, h, res, w * res, h * res, N))
print("occupied (val<100) : %7d  %6.2f %%" % (occ.sum(), 100.0 * occ.sum() / N))
print("free     (val>200) : %7d  %6.2f %%" % (free.sum(), 100.0 * free.sum() / N))
print("unknown  (val==128): %7d  %6.2f %%" % (unk.sum(), 100.0 * unk.sum() / N))
print("intermediate       : %7d  %6.2f %%" % (mid.sum(), 100.0 * mid.sum() / N))
print("free / occupied ratio : %.3f" % (free.sum() / max(1, occ.sum())))
r = runs(occ)
if r.size:
    rm = np.sort(r) * res
    print("occupied horizontal run lengths (m): n=%d median %.2f  p90 %.2f  p99 %.2f  max %.2f  mean %.2f"
          % (r.size, rm[len(rm) // 2], rm[int(0.90 * (len(rm) - 1))],
             rm[int(0.99 * (len(rm) - 1))], rm[-1], rm.mean()))
