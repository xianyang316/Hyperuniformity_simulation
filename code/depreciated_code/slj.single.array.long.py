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

# from tqdm import tqdm

def calc_sphere_volume(d):
    return 4/3 * np.pi * (d/2)**3

def create_PGN(d_center=2, d_polymer=1, n_strands=19, n_polymer=10):
    # load the spherical code file
    scode_file = f"SphericalCodes/pack.3.{n_strands}.txt"
    if not os.path.exists(scode_file):
        raise RuntimeError("spherical code file does not exist!")
    raw_codes = np.genfromtxt(scode_file)
    # reshape into [n_strands, (x, y, z)]
    coords = np.reshape(raw_codes, (-1, 3))
    # sanity check
    if coords.shape[0] != n_strands:
        raise RuntimeError("File contains wrong spherical code")
    # set the center of the particle
    p_center = [0, 0, 0]
    # create position array
    L = 2*d_polymer*n_polymer + d_center
    pos_array = np.zeros(shape=(n_strands*n_polymer + 1, 3))
    # populate the position array using list comprehension
    # because we instantiated with zeros, no need to set center
    pos_array[1:] = np.array([((d_center+d_polymer)/2)*s + d_polymer*n*s
                              for s in coords
                              for n in np.arange(n_polymer)])
    # create list of bonds and types
    # we'll use list comprehension to avoid nested for loops
    # we end up reshaping with (-1, 2) in order to return it
    # in the proper HOOMD format
    bond_group = np.array(
        [[[0, i*n_polymer+1],
          *[[i*n_polymer+j+1,
             i*n_polymer+j+2]
            for j in range(n_polymer-1)]]
         for i in range(n_strands)]).reshape(-1, 2)
    bond_types = ["bondAB", "bondBB"]
    bond_typeid = [item for sublist in [[0, *[1 for _ in range(n_polymer-1)]] for _ in range(n_strands)] for item in sublist]
    # this is actually pretty horrendous but necessary
    # since the * syntax doesn't work twice
    type_array = ["A", *["B" for _ in range(pos_array.shape[0]-1)]]
    diameter_array = [d_center, *[d_polymer for _ in range(pos_array.shape[0]-1)]]
    volume_array = [calc_sphere_volume(d) for d in diameter_array]
    mass_array = [4.3e9, *[1 for _ in range(pos_array.shape[0] - 1)]]
    return pos_array, diameter_array, mass_array, volume_array, type_array,\
        bond_group, bond_types, bond_typeid, L

hoomd.context.initialize("")

# specify information for PGN

d_center = 9.23
d_polymer = 1.0

n_polymer = 30
# n_strands = 19
n_strands = 130
pos_array, diameter_array, mass_array, volume_array, type_array,\
    bond_group, bond_types, bond_typeid, L = create_PGN(d_center, d_polymer, n_strands, n_polymer)
n_particles = len(diameter_array)

buffer = 5

unit_cell = hoomd.lattice.unitcell(
    N=pos_array.shape[0],
    a1=[L+buffer, 0, 0], a2=[0, L+buffer, 0], a3=[0, 0, L+buffer],
    dimensions=3,
    position=pos_array,
    diameter=diameter_array,
    mass=mass_array,
    type_name=type_array)
snap = unit_cell.get_snapshot()
snap.bonds.types = bond_types

snap.bonds.resize(bond_group.shape[0])

snap.bonds.group[:] = bond_group[:]
snap.bonds.typeid[:] = bond_typeid[:]

# create a single particle

n_replicates = 8
snap.replicate(n_replicates, n_replicates, n_replicates)

n_PGN = n_replicates**3
volume_PGN = np.sum(volume_array)

n_particles *= n_PGN
system = hoomd.init.read_snapshot(snap)

# nl = hoomd.md.nlist.cell()
nl = hoomd.md.nlist.tree(r_buff=1.0)
slj_wca = hoomd.md.pair.slj(r_cut=0, nlist=nl, d_max=d_center)
slj_wca.pair_coeff.set('A', 'A', sigma=1.0, epsilon=1.0, r_cut=2.**(1./6.))
slj_wca.pair_coeff.set('A', 'B', sigma=1.0, epsilon=1.0, r_cut=2.**(1./6.))
slj_wca.pair_coeff.set('B', 'B', sigma=1.0, epsilon=1.0, r_cut=2.**(1./6.))
slj_wca.set_params(mode="shift")
# trying per Josh's recommendation
# https://groups.google.com/g/hoomd-users/c/MTRPnF7zms4

# nl.reset_exclusions(exclusions=[])
hoomd.md.integrate.mode_standard(dt=0.0005)
fene = hoomd.md.bond.fene()
fene.bond_coeff.set("bondAB", k=30.0, r0=10, sigma=1.0, epsilon=1.0)
fene.bond_coeff.set("bondBB", k=30.0, r0=10, sigma=1.0, epsilon=1.0)

# bd = hoomd.md.integrate.brownian(group=hoomd.group.tags(1, n_particles-1), kT=0.5, seed=42)
# bd = hoomd.md.integrate.brownian(group=hoomd.group.type("B"), kT=0.5, seed=42)
# bd.set_gamma("A", 1.0)
# bd.set_gamma("B", 1.0)
langevin = hoomd.md.integrate.langevin(group=hoomd.group.type("B"), kT=0.5, seed=42)

hoomd.analyze.log(filename="log-slj.single.array.long.log",
                  quantities=["potential_energy", "temperature", "lx", "ly", "lz"],
                  period=100,
                  overwrite=True)
hoomd.dump.gsd("slj.single.array.long.gsd", period=1e5, group=hoomd.group.all(), overwrite=True, dynamic=["momentum"])

# thermalize
hoomd.run(1e7)
