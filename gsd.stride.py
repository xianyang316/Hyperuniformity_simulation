import gsd
import gsd.hoomd

with gsd.hoomd.open("../scripts/slj.hairy.gsd", "rb") as traj, gsd.hoomd.open("../scripts/slj.stride.gsd", "wb") as new_traj:
    print(len(traj))
    for frame in range(0, len(traj), 50):
        s = gsd.hoomd.Snapshot()
        s = traj[frame]
        new_traj.append(s)
