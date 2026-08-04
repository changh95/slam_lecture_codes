"""Independent arbiter: which trajectory (FAST-LIO2 vs KISS-SLAM) is consistent
with the raw /hesai/pandar scans?

Aggregate N consecutive scans into a common frame using each trajectory's poses
and count occupied voxels. A wrong trajectory smears surfaces across more
voxels. Identity (no motion) is the floor/ceiling sanity reference.
"""
import numpy as np, rosbag, sys

IN_DTYPE = np.dtype({"names":["x","y","z","intensity","timestamp","ring"],
                     "formats":["<f4","<f4","<f4","<f4","<f8","<u2"],
                     "offsets":[0,4,8,16,24,32],"itemsize":48})

def quat_to_R(qx,qy,qz,qw):
    n=np.sqrt(qx*qx+qy*qy+qz*qz+qw*qw); qx,qy,qz,qw=qx/n,qy/n,qz/n,qw/n
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)]])

def load_tum(p):
    out=[]
    for l in open(p):
        v=l.split()
        if len(v)!=8: continue
        t=float(v[0]); tr=np.array([float(v[1]),float(v[2]),float(v[3])])
        R=quat_to_R(*[float(x) for x in v[4:8]])
        out.append((t,R,tr))
    return out

def pose_at(traj, t):
    ts=np.array([p[0] for p in traj])
    i=int(np.argmin(np.abs(ts-t)))
    return traj[i], abs(ts[i]-t)

R_IL=np.array([[0.,-1.,0.],[-1.,0.,0.],[0.,0.,-1.]])
t_IL=np.array([-0.001,-0.00855,0.055])

START=int(sys.argv[1]); NSCAN=int(sys.argv[2]); VOX=float(sys.argv[3])
bag=rosbag.Bag('/data/exp14_basement_2.bag')
scans=[]
for i,(topic,msg,t) in enumerate(bag.read_messages(topics=['/hesai/pandar'])):
    if i<START: continue
    if i>=START+NSCAN: break
    a=np.frombuffer(msg.data,dtype=IN_DTYPE,count=msg.width*msg.height)
    p=np.stack([a['x'],a['y'],a['z']],axis=1).astype(np.float64)
    r=np.linalg.norm(p,axis=1)
    p=p[(r>0.5)&(r<20.0)]
    scans.append((msg.header.stamp.to_sec(), p))
bag.close()
npts=sum(len(p) for _,p in scans)
print("scans %d..%d  points used: %d  voxel %.2f m" % (START, START+NSCAN-1, npts, VOX))

fl=load_tum('/traj/fastlio_traj_tum.txt')
ks=load_tum('/traj/kiss_tum.txt')

def score(name, mode):
    allp=[]; maxdt=0.0
    for ts,p in scans:
        if mode=='identity':
            allp.append(p); continue
        if mode=='fastlio':
            (tt,R,tr),dt=pose_at(fl,ts+0.0997)   # FAST-LIO stamps == scan END
            Rw=R@R_IL; tw=R@t_IL+tr              # IMU pose -> LiDAR pose
        elif mode=='kiss':
            (tt,R,tr),dt=pose_at(ks,ts)
            Rw,tw=R,tr
        else:
            (tt,R,tr),dt=pose_at(ks,ts)
            Rw,tw=R.T,-R.T@tr
        maxdt=max(maxdt,dt)
        allp.append(p@Rw.T+tw)
    P=np.concatenate(allp,axis=0)
    keys=np.floor(P/VOX).astype(np.int64)
    keys-=keys.min(axis=0)
    k=(keys[:,0].astype(np.int64)<<42)|(keys[:,1].astype(np.int64)<<21)|keys[:,2]
    occ=np.unique(k).size
    print("  %-9s occupied voxels %8d   ratio-to-identity %.3f  (max stamp mismatch %.4f s)"
          % (name, occ, occ/score.ident if score.ident else 1.0, maxdt))
    return occ

score.ident=None
score.ident=score('identity','identity')
score('fastlio','fastlio')
score('kiss','kiss')
score('kiss_inv','kiss_inv')
