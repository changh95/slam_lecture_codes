import sys, math
p=sys.argv[1]
rows=[l.split() for l in open(p) if l.strip()]
xs=[(float(r[0]),float(r[1]),float(r[2]),float(r[3])) for r in rows]
L=0.0
for a,b in zip(xs,xs[1:]):
    L+=math.dist(a[1:],b[1:])
print("file:", p)
print("poses: %d   t: %.6f .. %.6f (%.3f s)" % (len(xs), xs[0][0], xs[-1][0], xs[-1][0]-xs[0][0]))
print("path length: %.3f m" % L)
print("start: %.4f %.4f %.4f" % xs[0][1:])
print("end  : %.4f %.4f %.4f" % xs[-1][1:])
print("end-start distance: %.4f m" % math.dist(xs[0][1:], xs[-1][1:]))
for i,n in enumerate("xyz",1):
    v=[r[i] for r in xs]
    print("%s range: %.3f .. %.3f (span %.3f)" % (n, min(v), max(v), max(v)-min(v)))
# max inter-frame jump (divergence detector)
jumps=sorted(math.dist(a[1:],b[1:]) for a,b in zip(xs,xs[1:]))
print("inter-frame step: median %.4f m, p99 %.4f m, max %.4f m" % (jumps[len(jumps)//2], jumps[int(0.99*len(jumps))], jumps[-1]))
dts=sorted(b[0]-a[0] for a,b in zip(xs,xs[1:]))
print("pose dt: min %.4f median %.4f max %.4f s" % (dts[0], dts[len(dts)//2], dts[-1]))
