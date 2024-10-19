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

hoomd.context.initialize("")

# specify information for PGN

d_center = 10.0
d_polymer = 1.0

n_polymer = 1
# n_strands = 19
n_strands = 1592
pos_array, bond_group, bond_types, bond_typeid, type_array, diameter_array, volume_array, L = create_PGN(d_center, d_polymer, n_strands, n_polymer, n_random=394)
# pos_array, bond_group, bond_types, bond_typeid, type_array, diameter_array, volume_array, L = create_PGN(d_center, d_polymer, n_strands, n_polymer, n_random=5)
n_particles = len(diameter_array)

buffer = 20

unit_cell = hoomd.lattice.unitcell(
    N=pos_array.shape[0],
    a1=[L+buffer, 0, 0], a2=[0, L+buffer, 0], a3=[0, 0, L+buffer],
    dimensions=3,
    position = pos_array,
    diameter=diameter_array,
    type_name=type_array)
snap = unit_cell.get_snapshot()
snap.bonds.types = bond_types

snap.bonds.resize(bond_group.shape[0])

snap.bonds.group[:] = bond_group[:]
snap.bonds.typeid[:] = bond_typeid[:]

# this will be cubed

n_replicates = 1
# snap.replicate(n_replicates, n_replicates, n_replicates)

n_PGN = n_replicates**3
volume_PGN = np.sum(volume_array)

n_particles *= n_PGN
system = hoomd.init.read_snapshot(snap)

# hoomd.dump.gsd("pgn.random-init.gsd", period=None, group=hoomd.group.all(), overwrite=True, dynamic=["momentum"])

nl = hoomd.md.nlist.cell()
# nl = hoomd.md.nlist.tree(r_buff=1.0)
# slj_wca = hoomd.md.pair.slj(r_cut=0, nlist=nl, d_max=d_center)
# slj_wca.pair_coeff.set('A', 'A', sigma=1.0, epsilon=1.0, r_cut=2.**(1./6.))
# slj_wca.pair_coeff.set('A', 'B', sigma=1.0, epsilon=1.0, r_cut=2.**(1./6.))
# # slj_wca.pair_coeff.set('B', 'B', sigma=1.0, epsilon=0.000000001, r_cut=2.**(1./6.))
# slj_wca.pair_coeff.set('B', 'B', sigma=1.0, epsilon=0.0, r_cut=2.**(1./6.))
# slj_wca.set_params(mode="shift")

lj = hoomd.md.pair.lj(r_cut=0, nlist=nl)
lj.pair_coeff.set("A", "A", epsilon=1.0, sigma=1.0)
lj.pair_coeff.set("A", "B", epsilon=1.0, sigma=1.0, r_cut=2.0**(1.0/6.0)*(d_center+d_polymer)/2)
poly_scale = 0.05
lj.pair_coeff.set("B", "B", epsilon=1.0, sigma=poly_scale*d_polymer, r_cut=2.0**(1.0/6.0)*poly_scale*d_polymer)
lj.set_params(mode="shift")

nl.reset_exclusions(exclusions=[])
hoomd.md.integrate.mode_standard(dt=0.0005)
harmonic = hoomd.md.bond.harmonic(name="polymer")
harmonic.bond_coeff.set("bondAB", k=100.0, r0=(d_center+d_polymer)/2)
harmonic.bond_coeff.set("bondBB", k=100.0, r0=d_polymer)
# fene = hoomd.md.bond.fene()
# fene.bond_coeff.set("bondAB", k=30.0, r0=10, sigma=1.0, epsilon=1.0)
# fene.bond_coeff.set("bondBB", k=30.0, r0=10, sigma=1.0, epsilon=0.1)

# bd = hoomd.md.integrate.brownian(group=hoomd.group.all(), kT=0.01, seed=42)
# bd = hoomd.md.integrate.brownian(group=hoomd.group.type("B"), kT=0.01, seed=42)
# bd.set_gamma("A", 1.0)
# bd.set_gamma("B", 1.0)
langevin = hoomd.md.integrate.langevin(group=hoomd.group.type("B"), kT=0.1, seed=42)

hoomd.analyze.log(filename="pgn.random-init.log",
                  quantities=["potential_energy", "temperature", "lx", "ly", "lz"],
                  period=100,
                  overwrite=True)
hoomd.dump.gsd("pgn.random-init.0.gsd", period=None, group=hoomd.group.all(), overwrite=True, dynamic=["momentum"])
hoomd.dump.gsd("pgn.random-init.gsd", period=1e4, group=hoomd.group.all(), overwrite=True, dynamic=["momentum"])

for poly_scale in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]:
    print(poly_scale)
    lj.pair_coeff.set("B", "B", epsilon=1.0, sigma=poly_scale*d_polymer, r_cut=2.0**(1.0/6.0)*poly_scale*d_polymer)
    hoomd.run(1e5)

hoomd.dump.gsd("pgn.random-init.end.gsd", period=None, group=hoomd.group.all(), overwrite=True, dynamic=["momentum"])
x = system.take_snapshot()
print(x.particles.position)


# for sigma in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:
#     slj_wca.pair_coeff.set('B', 'B', sigma=0.001, epsilon=1.0, r_cut=2.**(1./6.))
#     hoomd.run(1e5)

