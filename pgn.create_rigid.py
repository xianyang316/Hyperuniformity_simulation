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

# d_center, d_polymer, n_strands, coords,\
#     pos_array, type_array, diameter_array, volume_array = create_PGN_from_random_init("pgn.random-init.gsd", 1)
d_center, d_polymer, n_strands, coords, pos_array, type_array,\
    body_array, bond_group, bond_types, bond_typeid, diameter_array, volume_array = create_PGN_from_random_init("pgn.random-init.gsd", 1)

orientation_array = np.zeros(shape=[pos_array.shape[0], 4])
orientation_array[:, 0] = 1.0

L = 50.0
# m = 60e3
m = 1.0
# moi = 2/3*m*(d_center/2)**2
moi = m * (d_center/2)**2 + 2/3*(n_strands)*(d_center/2)**2
moi_array = np.zeros(shape=[pos_array.shape[0], 3])
moi_array[:, :] = m * (d_center/2)**2 + 2/3*(n_strands)*(d_center/2)**2

# uc = hoomd.lattice.unitcell(
#     # N=pos_array.shape[0],
#     N=1,
#     a1=[L, 0, 0], a2=[0, L, 0], a3=[0, 0, L],
#     dimensions=3,
#     # position=pos_array,
#     # diameter=diameter_array,
#     # orientation=orientation_array,
#     # moment_inertia=moi_array,
#     # type_name=type_array,
#     position=[[0, 0, 0]],
#     diameter=[d_center],
#     orientation=[[1, 0, 0, 0]],
#     moment_inertia=[[moi, moi, moi]],
#     type_name=["A"],
#     )

uc = hoomd.lattice.unitcell(
    N=1,
    a1=[L, 0, 0], a2=[0, L, 0], a3=[0, 0, L],
    dimensions=3,
    position=[[0, 0, 0]],
    diameter=[d_center],
    orientation=[[1.0, 0, 0, 0]],
    # mass=[1.0],
    moment_inertia=[[moi, moi, moi]],
    type_name=["A"],
    )

# system = hoomd.init.create_lattice(unitcell=uc, n=[5, 5, 5])
system = hoomd.init.create_lattice(unitcell=uc, n=[1, 1, 1])
system.particles.types.add("B")

rigid = hoomd.md.constrain.rigid()
rigid.set_param("A",
                types=["B"] * n_strands,
                positions=coords,
                )

rigid.create_bodies()

print([(p.body, p.tag) for p in system.particles])
sys.exit(0)

# uc = hoomd.lattice.unitcell(N = 1,
#                             a1 = [10.8, 0,   0],
#                             a2 = [0,    1.2, 0],
#                             a3 = [0,    0,   1.2],
#                             dimensions = 3,
#                             position = [[0,0,0]],
#                             type_name = ['A'],
#                             mass = [1.0],
#                             moment_inertia = [[0,
#                                                1/12*1.0*8**2,
#                                                1/12*1.0*8**2]],
#                             orientation = [[1, 0, 0, 0]]);
# system = hoomd.init.create_lattice(unitcell=uc, n=[2,18,18]);
# print(L)

# uc = hoomd.lattice.unitcell(N = 1,
#                             a1 = [L, 0,   0],
#                             a2 = [0,    L, 0],
#                             a3 = [0,    0,   L],
#                             dimensions = 3,
#                             position = [[0,0,0]],
#                             type_name = ['A'],
#                             mass = [1.0],
#                             moment_inertia = [[0,
#                                                1/12*1.0*8**2,
#                                                1/12*1.0*8**2]],
#                             orientation = [[1, 0, 0, 0]]);
# system = hoomd.init.create_lattice(unitcell=uc, n=[10, 10, 10]);

# system.particles.types.add('B');


# rigid = hoomd.md.constrain.rigid();
# rigid.set_param('A',
#                 types=['B']*8,
#                 positions=[(-4,0,0),(-3,0,0),(-2,0,0),(-1,0,0),
#                            (1,0,0),(2,0,0),(3,0,0),(4,0,0)]);
# rigid.create_bodies()

nl = hoomd.md.nlist.cell()
slj_wca = hoomd.md.pair.slj(r_cut=0, nlist=nl, d_max=d_center)
slj_wca.pair_coeff.set('A', 'A', sigma=1.0, epsilon=1.0, r_cut=2.**(1./6.))
slj_wca.pair_coeff.set('A', 'B', sigma=1.0, epsilon=1.0, r_cut=2.**(1./6.))
slj_wca.pair_coeff.set('B', 'B', sigma=1.0, epsilon=1.0, r_cut=2.**(1./6.))
slj_wca.set_params(mode="shift")
# lj = hoomd.md.pair.lj(r_cut=2**(1/6), nlist=nl)
# lj.set_params(mode='shift')

# lj.pair_coeff.set(['A', 'B'], ['A', 'B'], epsilon=1.0, sigma=1.0)

hoomd.md.integrate.mode_standard(dt=0.0005)
langevin = hoomd.md.integrate.langevin(group=hoomd.group.rigid_center(), kT=1.0, seed=42);
# langevin = hoomd.md.integrate.langevin(group=hoomd.group.type("C"), kT=0.1, seed=42)

hoomd.analyze.log(filename="pgn.rigid.log",
                  quantities=["potential_energy", "temperature", "lx", "ly", "lz", "translational_kinetic_energy", "rotational_kinetic_energy"],
                  period=100,
                  overwrite=True)
hoomd.dump.gsd("pgn.rigid.0.gsd", period=None, group=hoomd.group.all(), overwrite=True, dynamic=["momentum"])
hoomd.dump.gsd("pgn.rigid.gsd", period=1e4, group=hoomd.group.all(), overwrite=True, dynamic=["momentum"])

hoomd.run(1e5)
hoomd.dump.gsd("pgn.rigid.end.gsd", period=None, group=hoomd.group.all(), overwrite=True, dynamic=["momentum"])
