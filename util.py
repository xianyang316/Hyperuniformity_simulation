import sys
import os

import gsd
import gsd.hoomd

import numpy as np


def load_scode(n, skiprows=0):
    """load spherical code file for n spheres

    Args:
        n (int): spherical code number

    Returns:
        raw_codes (np.ndarray): numpy array
    """
    # load the spherical code file
    scode_file = f"SphericalCodes/pack.3.{n}.txt"
    if not os.path.exists(scode_file):
        raise RuntimeError("spherical code file does not exist!")
    raw_codes = np.genfromtxt(scode_file)
    return raw_codes


def load_random_sphere(fname):
    """load a nanoparticle core with randomly placed spheres

    Args:
        fname (str): name of the file to load

    Returns:
        frame (gsd frame): returns the final frame of the traj file
    """
    if not os.path.exists(fname):
        raise RuntimeError("pgn init file does not exist!")
    with gsd.hoomd.open(fname, mode="rb") as traj:
        f = traj[-1]
    return f


def calc_sphere_volume(d):
    """calculate the volume of a sphere
    Args:
        d: diameter

    Returns:
        volume
    """
    return 4/3 * np.pi * (d/2)**3

def create_PGN(d_center=2, d_polymer=1, n_strands=19, n_polymer=10, n_random=None):
    # load the spherical code file
    raw_codes = load_scode(n_strands)
    # reshape into [n_strands, (x, y, z)]
    coords = np.reshape(raw_codes, (-1, 3))
    # sanity check
    if coords.shape[0] != n_strands:
        raise RuntimeError("File contains wrong spherical code")
    coords /= np.linalg.norm(coords, axis=1)[:, np.newaxis]
    if n_random is not None:
        rand_idx = np.random.choice(np.arange(coords.shape[0]), n_random, replace=False)
        coords = coords[rand_idx]
    # set the center of the particle
    p_center = [0, 0, 0]
    # create position array
    L = 2*d_polymer*n_polymer + d_center
    if n_random is not None:
        n_strands = n_random
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
    return pos_array, bond_group, bond_types, bond_typeid, type_array, diameter_array, volume_array,  L


def create_PGN_from_random_init(fname, n_polymer):
    """create a PGN from a randomly initialized "core". Will be employing rigid bodies

    Args:
        fname (str): file name
        n_polymer (int): length of kuhn segments

    Returns:
        lots of stuff
    """
    gsd_frame = load_random_sphere(fname)
    # get the particle center
    # some brief error checking
    # take advantage of the mathematical set built into python
    if set(gsd_frame.particles.typeid.tolist()) != {0, 1}:
        raise ValueError("file must contain only two particle types: {0, 1}")
    if len(list(set(gsd_frame.particles.diameter.tolist()))) != 2:
        raise ValueError("only two distinct diameters allowed!")

    # get idx for center, polymers
    # this is hard-coded for center is type 0
    center_idx = np.argwhere(gsd_frame.particles.typeid == 0).ravel()
    if center_idx.shape[0] != 1:
        raise ValueError("currently only one central nanoparticle is allowed")
    polymer_idx = np.argwhere(gsd_frame.particles.typeid == 1).ravel()
    n_strands = polymer_idx.shape[0]

    # get the diameters
    d_center = gsd_frame.particles.diameter[center_idx][0]
    # already sanity checked for a single value
    d_polymer = list(set(gsd_frame.particles.diameter[polymer_idx].ravel().tolist()))[0]

    # define the position of the PGN
    # for sanity, re-center the vectors of the other particles
    coords = gsd_frame.particles.position[polymer_idx] - gsd_frame.particles.position[center_idx]
    # turn into unit vectors
    unit_coords = coords / np.linalg.norm(coords, axis=1)[:, np.newaxis]

    # return d_center, d_polymer, n_strands, coords

    # create position array
    L = 2*d_polymer*n_polymer + d_center

    pos_array = np.zeros(shape=(n_strands*n_polymer + 1, 3))

    # populate the position array using list comprehension
    # because we instantiated with zeros, no need to set center
    pos_array[1:] = np.array([((d_center+d_polymer)/2)*s + d_polymer*n*s
                              for s in unit_coords
                              for n in np.arange(n_polymer)])


    # create a 1D array of types...
    # this is actually pretty horrendous but necessary
    # since the * syntax doesn't work twice
    # create a list representing each polymer strand
    poly_list = ["B", *["P" for _ in range(n_polymer - 1)]]
    type_list = ["C", *[item for sublist in [poly_list for _ in range(n_strands)] for item in sublist]]
    # do the same thing for tags; be a little more explicit
    body_array = np.zeros(shape=(n_strands*n_polymer+1), dtype=np.int64)
    # create an array to hold the body tags for non-rigid polymers
    # poly_bodies = (((n_polymer-1) * np.arange(n_strands))[:, np.newaxis] + np.arange(0, n_polymer-1)[np.newaxis, :]) + 1
    # poly_bodies = np.hstack((np.zeros(shape=(poly_bodies.shape[0], 1), dtype=np.int64), poly_bodies))
    poly_bodies = -np.ones(shape=(n_strands, n_polymer), dtype=np.int64)
    poly_bodies[:, 0] = 0
    body_array[1:] = poly_bodies.ravel()[:]
    # now define bonds
    # bond_group = np.array(
    #     [[[0, i*n_polymer+1],
    #       *[[i*n_polymer+j+1,
    #          i*n_polymer+j+2]
    #         for j in range(n_polymer-1)]]
    #      for i in range(n_strands)]).reshape(-1, 2)
    bond_group = np.array(
        [[*[[i*n_polymer+j+1,
             i*n_polymer+j+2]
            for j in range(n_polymer-1)]]
         for i in range(n_strands)]).reshape(-1, 2)
    bond_types = ["polymer"]
    # bond_typeid = [item for sublist in [[0, *[1 for _ in range(n_polymer-1)]] for _ in range(n_strands)] for item in sublist]
    bond_typeid = [item for sublist in [[0 for _ in range(n_polymer-1)] for _ in range(n_strands)] for item in sublist]
    diameter_array = [d_center, *[d_polymer for _ in range(pos_array.shape[0]-1)]]
    volume_array = [calc_sphere_volume(d) for d in diameter_array]

    return d_center, d_polymer, n_strands, coords, pos_array, type_list, \
        body_array, bond_group, bond_types, bond_typeid, diameter_array, volume_array


def create_PGN_from_scode(d_center=2, d_polymer=1, n_strands=19, n_polymer=10, n_random=None):
    # load the spherical code file
    raw_codes = load_scode(n_strands)
    # reshape into [n_strands, (x, y, z)]
    coords = np.reshape(raw_codes, (-1, 3))
    # sanity check
    if coords.shape[0] != n_strands:
        raise RuntimeError("File contains wrong spherical code")
    coords /= np.linalg.norm(coords, axis=1)[:, np.newaxis]
    if n_random is not None:
        coords = coords[np.random.randint(0, coords.shape[0], n_random)]
    # set the center of the particle
    p_center = [0, 0, 0]
    # create position array
    L = 2*d_polymer*n_polymer + d_center
    if n_random is not None:
        n_strands = n_random
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
    return pos_array, bond_group, bond_types, bond_typeid, type_array, diameter_array, volume_array,  L
