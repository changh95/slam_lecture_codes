#!/usr/bin/env python3
"""Per-scan odometry time, queue depth and real-time factor from a glim_rosbag run with -p debug:=true.

Mount a host dir on the container /tmp so glim_log.log and glim_odom.log land there.
usage: odom_timing.py <dir with glim_log.log and glim_odom.log>
"""
import re,sys,datetime,numpy as np
ts=lambda s: datetime.datetime.strptime(s,'%Y-%m-%d %H:%M:%S.%f').timestamp()
pts=[];ins=[];upd=[];ev={}
import os
D=sys.argv[1]
lines=[l for f in ('glim_log.log','glim_odom.log') for l in open(os.path.join(D,f),errors='replace')]
for l in lines:
    m=re.match(r'\[(\S+ \S+)\] \[(\w+)\] \[(\w+)\] (.*)',l)
    if not m: continue
    t=ts(m.group(1)); msg=m.group(4)
    if msg.startswith('points: '): pts.append((t,float(msg[8:])))
    elif msg.startswith('insert_frame points='): ins.append(t)
    elif msg=='frames updated': upd.append(t)
    elif 'waiting for odometry' in msg: ev['bag_end']=t
    elif 'waiting for local mapping' in msg: ev['odom_done']=t
    elif 'initial IMU state estimation result' in msg: ev['init']=t
ins.sort(); upd.sort(); pts.sort()
print(f'points cb {len(pts)}, odom insert {len(ins)}, frames updated {len(upd)}')
n=min(len(ins),len(upd))
# per-scan odom time: pair each 'frames updated' with the preceding insert
ins=np.array(ins); upd=np.array(upd)
k=np.searchsorted(ins,upd)-1; dur=(upd-ins[k])*1000
print(f'per-scan odometry ms: median {np.median(dur):.1f} mean {dur.mean():.1f} p95 {np.percentile(dur,95):.1f} max {dur.max():.1f}')
p=np.array(pts); m=min(len(p),len(ins))
lag=ins[:m]-p[:m,0]
q=np.array([np.sum((p[:m,0]<=t))-np.sum(ins<=t) for t in p[:m,0]])  # frames queued when each scan arrived
print(f'queue lag (reader insert -> odom start) s: median {np.median(lag):.2f} p95 {np.percentile(lag,95):.2f} max {lag.max():.2f} last {lag[-1]:.2f}')
thirds=np.array_split(q,3); print('queue depth (frames) per third of slice: ', [int(x.mean()) for x in thirds], 'max', q.max())
bag_span=p[-1,1]-p[0,1]
wall=ev.get('odom_done',upd[-1])-p[0,0]
print(f'bag span {bag_span:.1f} s, wall first scan -> odometry done {wall:.1f} s, real-time factor {bag_span/wall:.2f}x')
if 'bag_end' in ev and 'odom_done' in ev: print(f"backlog drain after bag end: {ev['odom_done']-ev['bag_end']:.1f} s")
iv=np.diff(upd)*1000
print(f'odometry output interval (drives viewer updates) ms: median {np.median(iv):.0f} p95 {np.percentile(iv,95):.0f} p99 {np.percentile(iv,99):.0f} max {iv.max():.0f}; >300 ms stalls: {(iv>300).sum()}')
