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
    pos_array[1:] = np.array([(((d_center+d_polymer)/2)*s) + d_polymer*n*s
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
    mass_array = [10000, *[1 for _ in range(pos_array.shape[0] - 1)]]
    return pos_array, bond_group, bond_types, bond_typeid, type_array, diameter_array, mass_array, volume_array, L

hoomd.context.initialize("")

# specify information for PGN

d_center = 9.23
# d_center = 5.0
d_polymer = 1.0

n_polymer = 15
# n_strands = 19
n_strands = 130
pos_array, bond_group, bond_types, bond_typeid, type_array, diameter_array, mass_array, volume_array, L = create_PGN(d_center, d_polymer, n_strands, n_polymer)
n_particles = len(diameter_array)

buffer = 30

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
print(snap.bonds.typeid)

# print(snap.particles.typeid[:])

# create a single particle

n_PGN = 1
volume_PGN = np.sum(volume_array)

n_particles *= n_PGN
system = hoomd.init.read_snapshot(snap)

nl = hoomd.md.nlist.cell()
# trying per Josh's recommendation
# https://groups.google.com/g/hoomd-users/c/MTRPnF7zms4
# lj_wca = hoomd.md.pair.lj(r_cut=0, nlist=nl)
# sigma_AA = d_center
# sigma_AB = (d_center + d_polymer)/2
# sigma_BB = d_polymer
# lj_wca.pair_coeff.set('A', 'A', sigma=sigma_AA, epsilon=1.0, r_cut=2.**(1./6.)*d_center)
# lj_wca.pair_coeff.set('A', 'B', sigma=sigma_AB, epsilon=1.0, r_cut=2.**(1./6.)*(d_center+d_polymer)/2)
# lj_wca.pair_coeff.set('B', 'B', sigma=sigma_BB, epsilon=1.0, r_cut=2.**(1./6.)*d_polymer)
# lj_wca.set_params(mode="shift")
slj_wca = hoomd.md.pair.slj(r_cut=0, nlist=nl, d_max=d_center)
slj_wca.pair_coeff.set('A', 'A', sigma=1.0, epsilon=1.0, r_cut=2.**(1./6.))
slj_wca.pair_coeff.set('A', 'B', sigma=1.0, epsilon=1.0, r_cut=2.**(1./6.))
slj_wca.pair_coeff.set('B', 'B', sigma=1.0, epsilon=1.0, r_cut=2.**(1./6.))
slj_wca.set_params(mode="shift")

# nl.reset_exclusions(exclusions=[])
hoomd.md.integrate.mode_standard(dt=0.0005)
# harmonic = hoomd.md.bond.harmonic()
# harmonic.bond_coeff.set("bondAB", k=100.0, r0=(d_center+d_polymer)/2)
# harmonic.bond_coeff.set("bondBB", k=100.0, r0=d_polymer)
fene = hoomd.md.bond.fene()
fene.bond_coeff.set("bondAB", k=30.0, r0=10, sigma=1.0, epsilon=1.0)
fene.bond_coeff.set("bondBB", k=30.0, r0=10, sigma=1.0, epsilon=1.0)

# bd = hoomd.md.integrate.brownian(group=hoomd.group.all(), kT=0.05, seed=42)
langevin = hoomd.md.integrate.langevin(group=hoomd.group.all(), kT=0.5, seed=42)
# bd = hoomd.md.integrate.brownian(group=hoomd.group.tags(1, n_particles-1), kT=0.5, seed=42)
# bd = hoomd.md.integrate.brownian(group=hoomd.group.type("B"), kT=0.1, seed=42)
# bd.set_gamma("A", 50.0)
# bd.set_gamma("B", 1.0)

hoomd.analyze.log(filename="log-dpd.single.log",
                  quantities=["potential_energy", "temperature", "lx", "ly", "lz"],
                  period=100,
                  overwrite=True)
hoomd.dump.gsd("slj.single.fene.mass.gsd", period=1e4, group=hoomd.group.all(), overwrite=True, dynamic=["momentum"])
# hoomd.dump.gsd("slj.single.fene.gsd", period=1, group=hoomd.group.all(), overwrite=True, dynamic=["momentum"])

# thermalize
hoomd.run(1e4)
nl.tune()
hoomd.run(1e7)
sys.exit(0)
# fast compression
compress_period = 1e5
compress_steps = 10e6
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
target_phi = 0.4
current_volume = system.box.get_volume()
current_phi = n_PGN * volume_PGN / current_volume
while current_phi < target_phi:
    current_phi *= 1.05
    current_volume = n_PGN * volume_PGN / current_phi
    current_L = np.power(current_volume, (1/3))
    hoomd.update.box_resize(L=current_L, period=None)
    hoomd.run(1e6, quiet=False)

hoomd.run(10e6)
