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

d_center = 9.23
n_polymer = 15
n_strands = 130
d_polymer = 1.0

# load in dense gsd
with gsd.hoomd.open("./init-random.gsd", "rb") as traj:
    # get the final frame
    c_sphere_frame = traj[-1]

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
# print(d_max)
# sys.exit(0)

# how do we go about creating the lists
# start with positions
# we could "get cute" but let's just be simple?
pos_list = list()
type_list = list()
for c_position in c_sphere_frame.particles.position:
    # all we should need to do is to take the position
    # of the circumsphere and add the position of each
    # particle in the PGN
    pos_list.extend((np.asarray(PGN_frame.particles.position) + c_position).tolist())
pos_array = np.asarray(np.asarray(pos_list))

for PGN_idx, _ in enumerate(c_sphere_frame.particles.position):
    type_list.extend(PGN_frame.particles.typeid)

diameter_list = list()
for c_diameter in c_sphere_frame.particles.diameter:
    diameter_list.extend(PGN_frame.particles.diameter.tolist())
diameter_array = np.asarray(diameter_list)
volume_array = [calc_sphere_volume(d) for d in diameter_array]
volume_PGN = np.sum(volume_array)

bond_group_list = list()
bond_typeid_list = list()
n_PGN_beads = PGN_frame.particles.position.shape[0]
for PGN_idx, _ in enumerate(c_sphere_frame.particles.position):
    bond_group_list.extend((np.asarray(PGN_frame.bonds.group) + n_PGN_beads*PGN_idx).tolist())
    bond_typeid_list.extend(PGN_frame.bonds.typeid.tolist())

bond_types_list = ["bondAB", "bondBB"]

# now we should be able to initialize the snapshot
box = 1.5*c_sphere_frame.configuration.box

snap = hoomd.data.make_snapshot(
    N=pos_array.shape[0],
    box=hoomd.data.boxdim(*box),
    particle_types=["A", "B"],
    bond_types=bond_types_list)
# populate the snapshot
snap.particles.position[:] = pos_array[:]
snap.particles.diameter[:] = diameter_array[:]
snap.particles.typeid[:] = type_list[:]
snap.bonds.resize(len(bond_group_list))
snap.bonds.group[:] = bond_group_list[:]
snap.bonds.typeid[:] = bond_typeid_list[:]

system = hoomd.init.read_snapshot(snap)

# try SLJ
nl = hoomd.md.nlist.tree(r_buff=1.0)
# trying per Josh's recommendation
# https://groups.google.com/g/hoomd-users/c/MTRPnF7zms4
lj_wca = hoomd.md.pair.lj(r_cut=0, nlist=nl)
lj_wca.pair_coeff.set('A', 'A', sigma=(d_center), epsilon=1.0, r_cut=2.**(1./6.) * (d_center))
lj_wca.pair_coeff.set('A', 'B', sigma=((d_center+d_polymer)/2), epsilon=1.0, r_cut=2.**(1./6.) * ((d_center+d_polymer)/2))
lj_wca.pair_coeff.set('B', 'B', sigma=(d_polymer), epsilon=1.0, r_cut=3.0 * (d_polymer))

hoomd.md.integrate.mode_standard(dt=0.0001)
# hoomd.md.integrate.mode_standard(dt=0.01)
# harmonic = hoomd.md.bond.harmonic()
# harmonic.bond_coeff.set("bondAB", k=100.0, r0=(d_center+d_polymer)/2)
# harmonic.bond_coeff.set("bondBB", k=100.0, r0=d_polymer)
fene = hoomd.md.bond.fene()
fene.bond_coeff.set("bondAB", k=100.0, r0=(d_center+d_polymer)/2, sigma=d_polymer, epsilon=1.0)
fene.bond_coeff.set("bondBB", k=30.0, r0=1.5*d_polymer, sigma=d_polymer, epsilon=1.0)

bd = hoomd.md.integrate.brownian(group=hoomd.group.all(), kT=0.5, seed=42)
bd.set_gamma("A", 1.0)
bd.set_gamma("B", 1.0)

hoomd.analyze.log(filename="log.fene.hairy.log",
                  quantities=["potential_energy", "temperature", "lx", "ly", "lz"],
                  period=100,
                  overwrite=True)
hoomd.dump.gsd("slj.fene.hairy.gsd", period=1e6, group=hoomd.group.all(), overwrite=True, dynamic=["momentum"])
# hoomd.dump.dcd("dpd.hairy.dcd", period=1e4, group=hoomd.group.all(), overwrite=True, unwrap_full=True)

# thermalize
hoomd.run(1e6)
nl.tune()
hoomd.run(5e7)

