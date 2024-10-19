"""
Self-assembly of Brownian PGNs
"""
import sys
# imports
import hoomd
import hoomd.md

import gsd
import gsd.hoomd

import freud

import numpy as np

from tqdm import tqdm

def calc_sphere_volume(d):
    return 4/3 * np.pi * (d/2)**3

def create_PGN(d_center=2, d_polymer=1, n_polymer=10):
    raw_codes = np.genfromtxt("SphericalCodes/pack.3.19.txt")
    coords = np.reshape(raw_codes, (-1, 3))
    p_center = [0, 0, 0]
    # create position array
    L = 2*d_polymer*n_polymer + d_center
    pos_array = np.zeros(shape=(len(coords)*n_polymer + 1, 3))
    pos_array[1:] = np.array([((d_center+d_polymer)/2)*s + d_polymer*n*s
                              for s in coords
                              for n in np.arange(n_polymer)])
#     pos_array[1:] = coords
    type_array = ["A", *["B" for _ in range(pos_array.shape[0]-1)]]
    diameter_array = [d_center, *[d_polymer for _ in range(pos_array.shape[0]-1)]]
    volume_array = [calc_sphere_volume(d) for d in diameter_array]
    return pos_array, type_array, diameter_array, volume_array, L

hoomd.context.initialize("")

# specify information for PGN

d_center = 2.0
d_polymer = 1.0

n_polymer = 25
n_strands = 19
pos_array, type_array, diameter_array, volume_array, L = create_PGN(d_center, d_polymer, n_polymer)
n_particles = len(diameter_array)

unit_cell = hoomd.lattice.unitcell(
    N=pos_array.shape[0],
    a1=[L+5, 0, 0], a2=[0, L+5, 0], a3=[0, 0, L+5],
    dimensions=3,
    position = pos_array,
    diameter=diameter_array,
    type_name=type_array)
snap = unit_cell.get_snapshot()

snap.bonds.resize(len(pos_array) - 1)

bonds = np.zeros(shape=(len(pos_array) - 1, 2), dtype=np.int64)
btypes = list()
# this "works"
# n_strands = 13
# n_polymer = 10
for i in range(n_strands):
    bonds[i*n_polymer] = [0, i*n_polymer + 1]
    btypes.append("bondAB")
    for j in range(n_polymer-1):
        bonds[i*n_polymer + j+1] = [i*n_polymer+j+1, i*n_polymer+j+2]
        btypes.append("bondBB")
# print(bonds.reshape(n_polymer, -1, 2))
for i, bond in enumerate(bonds):
    snap.bonds.group[i] = bond
snap.bonds.types = btypes

# this will be cubed
n_replicates = 7
snap.replicate(n_replicates, n_replicates, n_replicates)

n_PGN = n_replicates**3
volume_PGN = np.sum(volume_array)

n_particles *= n_PGN
system = hoomd.init.read_snapshot(snap)

nl = hoomd.md.nlist.cell()
# slj = hoomd.md.pair.slj(r_cut=2.5*d_center, nlist=nl, d_max=np.nanmax([d_center, d_polymer]))
# slj.pair_coeff.set("A", "A", epsilon=1.0, sigma=1.0, r_cut=2**(1/6)*(d_center/2))
# slj.pair_coeff.set("A", "B", epsilon=1.0, sigma=1.0, r_cut=2**(1/6)*(d_center/2))
# slj.pair_coeff.set("B", "B", epsilon=1.0, sigma=0.5, r_cut=2**(1/6)*(d_polymer/2))
# trying per Josh's recommendation
# https://groups.google.com/g/hoomd-users/c/MTRPnF7zms4
lj_wca = hoomd.md.pair.lj(r_cut=0, nlist=nl)
lj_wca.pair_coeff.set('A', 'A', epsilon=(d_center/2), sigma=1.0, r_cut=2.**(1./6.) * (d_center/2))
lj_wca.pair_coeff.set('A', 'B', epsilon=((d_center+d_polymer)/2), sigma=1.0, r_cut=2.**(1./6.) * ((d_center+d_polymer)/2))
lj_wca.pair_coeff.set('B', 'B', epsilon=(d_polymer/2), sigma=1.0, r_cut=2.5 * (d_polymer/2))
lj_wca.set_params(mode="shift")
# slj.pair_coeff.set("B", "B", epsilon=0.25, sigma=(d_center/2), r_cut=2.5*d_center)

# hoomd.md.integrate.mode_standard(dt=0.005)

nl.reset_exclusions(exclusions=[])
hoomd.md.integrate.mode_standard(dt=0.0005)
harmonic = hoomd.md.bond.harmonic()
harmonic.bond_coeff.set("bondAB", k=100.0, r0=1.0)
harmonic.bond_coeff.set("bondBB", k=100.0, r0=1.0)

# bd = hoomd.md.integrate.brownian(group=hoomd.group.all(), kT=1.0, seed=42)
bd = hoomd.md.integrate.langevin(group=hoomd.group.all(), kT=1.0, seed=42)
bd.set_gamma("A", 1.0)
bd.set_gamma("B", 1.0)

hoomd.analyze.log(filename="log-output.hairy.log",
                  quantities=["potential_energy", "temperature", "lx", "ly", "lz"],
                  period=100,
                  overwrite=True)
hoomd.dump.gsd("trajectory.hairy.gsd", period=1e5, group=hoomd.group.all(), overwrite=True, dynamic=["momentum"])

# thermalize
hoomd.run(1e6)
# fast compression
compress_period = 1e5
compress_steps = 10e6
target_phi = 0.20
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

# slow compression
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
