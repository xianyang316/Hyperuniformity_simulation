"""
use hard-sphere lj to init a random system
"""

import sys
import os

import hoomd
import hoomd.md

import gsd
import gsd.hoomd

import numpy as np

def calc_sphere_volume(d):
    return 4/3 * np.pi * (d/2)**3

hoomd.context.initialize("")

d_polymer = 1.0
# load in randomized nanoparticle
with gsd.hoomd.open("./slj.single.gsd", "rb") as traj:
    # get the final frame
    PGN_frame = traj[-1]

# calculate circumsphere
ptypes = PGN_frame.particles.typeid
pos = PGN_frame.particles.position
p_center = pos[ptypes==0][0]
p_poly = pos[ptypes==1]
r_ij = p_poly - p_center
# print(r_ij)
# print(np.nanmax(np.linalg.norm(r_ij, axis=1)))
r_max = np.nanmax(np.linalg.norm(r_ij, axis=1)) + d_polymer/2
d_max = 2 * r_max
d_center = d_max

unit_cell = hoomd.lattice.unitcell(
    N=1,
    a1=[d_center+2, 0, 0], a2=[0, d_center+2, 0], a3=[0, 0, d_center+2],
    dimensions=3,
    position=[[0, 0, 0]],
    diameter=[d_center],
    type_name=["A"])
snap = unit_cell.get_snapshot()
snap.particles.typeid[:] = 0
snap.particles.diameter[:] = d_center

n_replicates = 5
snap.replicate(n_replicates,
               n_replicates, n_replicates)

n_PGN = n_replicates**3
volume_PGN = calc_sphere_volume(d_center)

n_particles = n_PGN
system = hoomd.init.read_snapshot(snap)

nl = hoomd.md.nlist.cell()

lj_wca = hoomd.md.pair.lj(r_cut=2.**(1./6.) * (d_center/2), nlist=nl)
lj_wca.pair_coeff.set('A', 'A', epsilon=1.0, sigma=(d_center), r_cut=2.**(1./6.) * (d_center))

lj_wca.set_params(mode="shift")

hoomd.md.integrate.mode_standard(dt=0.001)

bd = hoomd.md.integrate.brownian(group=hoomd.group.all(), kT=0.5, seed=42)
bd.set_gamma("A", 1.0)

hoomd.analyze.log(filename="log-wca.init.log",
                  quantities=["potential_energy", "temperature", "lx", "ly", "lz"],
                  period=100,
                  overwrite=True)
hoomd.dump.gsd("init-random.gsd", period=100, group=hoomd.group.all(), overwrite=True, dynamic=["momentum"])

hoomd.run(1e5)
# sys.exit(0)
# nl.tune()
# fast compression
compress_period = 1e5
compress_steps = 5e6
target_phi = 0.7
target_volume = n_PGN * volume_PGN / target_phi
target_L = np.power(target_volume, (1/3))
# determine the current value of L
l_list = list()
for L in ["Lx", "Ly", "Lz"]:
    l_list.append(system.box.__getattribute__(L))
if not all(l_list):
    raise RuntimeError("box is not cubic!")
current_L = l_list[0]
L_variant = hoomd.variant.linear_interp([(0, current_L), (compress_steps, target_L)], zero=hoomd.get_step())
# set the box size updater
hoomd.update.box_resize(L=L_variant, period=compress_period)
hoomd.run(compress_steps)
hoomd.run(1e6)
