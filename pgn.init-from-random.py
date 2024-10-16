"""
Self-assembly of Brownian PGNs
"""
import sys
import os
# imports
import hoomd
import hoomd.md

import gsd
import gsd.hoomd

# import freud

import numpy as np

from tqdm import tqdm

from util import calc_sphere_volume
from util import create_PGN
from util import load_random_sphere
from util import create_PGN_from_random_init

hoomd.context.initialize("")

f = load_random_sphere("pgn.random-init.gsd")

d_center, d_polymer, n_strands, coords, pos_array, type_array,\
    body_array, bond_group, bond_types, bond_typeid, diameter_array, volume_array = create_PGN_from_random_init("pgn.random-init.gsd", 20)

orientation_array = np.zeros(shape=[pos_array.shape[0], 4], dtype=np.float64)
orientation_array[:, 0] = 1.0

L = 50.0
m = 2.94e3
# m = 1.0
moi = m * (d_center/2)**2 + 2/3*(n_strands)*(d_center/2)**2
# moi_array = np.zeros(shape=[pos_array.shape[0], 3], dtype=np.float64)
mass_array = np.ones(shape=[pos_array.shape[0]], dtype=np.float64)
mass_array[type_array=="C"] = m
moi_array = np.ones(shape=[pos_array.shape[0], 3], dtype=np.float64)
moi_array[type_array=="C"] = moi
moi_array[type_array=="B"] = moi

uc = hoomd.lattice.unitcell(
    N=pos_array.shape[0],
    a1=[L, 0, 0], a2=[0, L, 0], a3=[0, 0, L],
    dimensions=3,
    position=pos_array,
    diameter=diameter_array,
    mass=mass_array,
    orientation=orientation_array,
    moment_inertia=moi_array,
    type_name=type_array,
)

n_particles = len(diameter_array)

snap = uc.get_snapshot()
snap.bonds.types = bond_types
snap.bonds.resize(bond_group.shape[0])
snap.bonds.group[:] = bond_group[:]
snap.bonds.typeid[:] = bond_typeid[:]
snap.particles.body[:] = body_array[:]

n_replicates = 5
snap.replicate(n_replicates, n_replicates, n_replicates)

n_PGN = n_replicates**3
volume_PGN = np.sum(volume_array)

n_particles *= n_PGN

system = hoomd.init.read_snapshot(snap)

rigid = hoomd.md.constrain.rigid()
rigid.set_param("C",
                types=["B"] * n_strands,
                positions=coords,
                )


rigid.validate_bodies()

n_center = hoomd.md.nlist.cell()
nl = hoomd.md.nlist.tree(r_buff=1.0)
slj_wca = hoomd.md.pair.slj(r_cut=0, nlist=nl, d_max=d_center)
slj_wca.pair_coeff.set('C', 'C', sigma=1.0, epsilon=1.0, r_cut=2.**(1./6.))
slj_wca.pair_coeff.set('C', 'B', sigma=1.0, epsilon=1.0, r_cut=2.**(1./6.))
slj_wca.pair_coeff.set('C', 'P', sigma=1.0, epsilon=1.0, r_cut=2.**(1./6.))
slj_wca.pair_coeff.set('B', 'P', sigma=1.0, epsilon=1.0, r_cut=2.**(1./6.))
slj_wca.pair_coeff.set('B', 'B', sigma=1.0, epsilon=1.0, r_cut=2.**(1./6.))
slj_wca.pair_coeff.set('P', 'P', sigma=1.0, epsilon=1.0, r_cut=2.**(1./6.))
slj_wca.set_params(mode="shift")

# slj_center = hoomd.md.pair.slj(r_cut=0, nlist=n_center, d_max=d_center)
# slj_center.pair_coeff.set('C', 'C', sigma=1.0, epsilon=1.0, r_cut=2.**(1./6.))
# slj_center.pair_coeff.set('C', 'B', sigma=1.0, epsilon=0.0, r_cut=0.0)
# slj_center.pair_coeff.set('C', 'P', sigma=1.0, epsilon=0.0, r_cut=0.0)
# slj_center.pair_coeff.set('B', 'P', sigma=1.0, epsilon=0.0, r_cut=0.0)
# slj_center.pair_coeff.set('B', 'B', sigma=1.0, epsilon=0.0, r_cut=0.0)
# slj_center.pair_coeff.set('P', 'P', sigma=1.0, epsilon=0.0, r_cut=0.0)
# slj_center.set_params(mode="shift")

# slj_wca = hoomd.md.pair.slj(r_cut=0, nlist=nl, d_max=d_center)
# slj_wca.pair_coeff.set('C', 'C', sigma=1.0, epsilon=0.0, r_cut=0.0)
# slj_wca.pair_coeff.set('C', 'B', sigma=1.0, epsilon=1.0, r_cut=2.**(1./6.))
# slj_wca.pair_coeff.set('C', 'P', sigma=1.0, epsilon=1.0, r_cut=2.**(1./6.))
# slj_wca.pair_coeff.set('B', 'P', sigma=1.0, epsilon=1.0, r_cut=2.**(1./6.))
# slj_wca.pair_coeff.set('B', 'B', sigma=1.0, epsilon=1.0, r_cut=2.**(1./6.))
# slj_wca.pair_coeff.set('P', 'P', sigma=1.0, epsilon=1.0, r_cut=2.**(1./6.))
# slj_wca.set_params(mode="shift")

fene = hoomd.md.bond.fene()
fene.bond_coeff.set("polymer", k=30.0, r0=10, sigma=1.0, epsilon=1.0)

hoomd.md.integrate.mode_standard(dt=0.001)
rigid = hoomd.group.rigid_center();
langevin_rigid = hoomd.md.integrate.langevin(group=rigid, kT=1.0, seed=561);
langevin_polymer = hoomd.md.integrate.langevin(group=hoomd.group.type("P"), kT=1.0, seed=991);
# langevin = hoomd.md.integrate.langevin(group=hoomd.group.type("C"), kT=0.1, seed=42)

hoomd.analyze.log(filename="pgn.rigid.3.log",
                  quantities=["potential_energy", "temperature", "lx", "ly", "lz", "translational_kinetic_energy", "rotational_kinetic_energy"],
                  period=1000,
                  overwrite=True)
hoomd.dump.gsd("pgn.rigid.3.gsd", period=1e6, group=hoomd.group.all(), overwrite=True, dynamic=["momentum"])
hoomd.run(1e6)

nl.tune()
hoomd.run(1e7)
# fast compression
compress_period = 1e5
compress_steps = 1e7
target_phi = 0.30
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

# slower compression
target_phi = 0.55
current_volume = system.box.get_volume()
current_phi = n_PGN * volume_PGN / current_volume
while current_phi < target_phi:
    current_phi *= 1.05
    current_volume = n_PGN * volume_PGN / current_phi
    current_L = np.power(current_volume, (1/3))
    hoomd.update.box_resize(L=current_L, period=None)
    hoomd.run(1e6, quiet=False)

hoomd.run(4e7)
