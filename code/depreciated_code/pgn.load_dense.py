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

d_center, d_polymer, n_strands, coords, pos_array, type_array,\
    body_array, bond_group, bond_types, bond_typeid, diameter_array, volume_array = create_PGN_from_random_init("pgn.random-init.gsd", 20)

hoomd.init.read_gsd("pgn.rigid.3.out.gsd")

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

hoomd.analyze.log(filename="pgn.reload.log",
                  quantities=["potential_energy", "temperature", "lx", "ly", "lz", "translational_kinetic_energy", "rotational_kinetic_energy"],
                  period=1000,
                  overwrite=True)
hoomd.dump.gsd("pgn.reload.gsd", period=1e6, group=hoomd.group.all(), overwrite=True, dynamic=["momentum"])
hoomd.run(1e6)

nl.tune()
hoomd.run(1e7)
sys.exit(0)
