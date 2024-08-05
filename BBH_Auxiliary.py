#!/usr/bin/env python
## HEY
import numpy as np
import os
if os.environ.get('DISPLAY','') == '':
   print('No display found. Using non-interactive Agg backend')
   import matplotlib as mpl
   mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import InterpolatedUnivariateSpline

import surfinBH
import NRSur7dq2
from NRSur7dq2 import harmonics

from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d import proj3d
import matplotlib.animation as animation
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib.colors import SymLogNorm, Normalize
from matplotlib.colors import LinearSegmentedColormap

#----------------------------------------------------------------------------
def quat_between_vecs(u, v):
    """ Computes quaternion for rotation from one vector to another """

    def normalize(v):
        return v/np.linalg.norm(v)

    def quat_fromaxisangle(axis, angle):
        half_sin = np.sin(0.5 * angle)
        half_cos = np.cos(0.5 * angle)
        return np.array([half_cos, \
            half_sin*axis[0], \
            half_sin*axis[1], \
            half_sin*axis[2]])

    axis = normalize(np.cross(u, v))
    angle = np.arccos(np.dot(normalize(u), normalize(v)))
    return quat_fromaxisangle(axis, angle)

#----------------------------------------------------------------------------
def spline_interp(newX, oldX, oldY, allowExtrapolation=False):
    """ Interpolates using splnes.
        If allowExtrapolation=True, extrapolates to zero.
    """
    if len(oldY) != len(oldX):
        raise Exception('Lengths dont match.')

    if not allowExtrapolation:
        if np.min(newX) - np.min(oldX) < -1e-5 \
                or np.max(newX) > np.max(oldX) > 1e-5:

            print(np.min(newX), np.min(oldX), np.max(newX), np.max(oldX))
            print(np.min(newX) < np.min(oldX))
            print(np.max(newX) > np.max(oldX))
            raise Exception('Trying to extrapolate, but '\
                'allowExtrapolation=False')

    if not np.all(np.diff(oldX) > 0):
        raise Exception('oldX must have increasing values')

    # returns 0 when extrapolating
    newY = InterpolatedUnivariateSpline(oldX, oldY, ext=1)(newX)
    return newY

#----------------------------------------------------------------------------
def get_trajectory(separation, quat_nrsur, orbphase_nrsur, bh_label):
    """ Gets trajectory of a component BH in a binary given the separation,
    the coprecessing frame quaternion and orbital phase in the coprecessing
    frame.
    """
    if bh_label == 'A':
        offset = 0
    else:
        offset = np.pi

    x_copr = separation * np.cos(orbphase_nrsur+offset)
    y_copr = separation * np.sin(orbphase_nrsur+offset)
    z_copr = np.zeros(len(x_copr))

    Bh_traj_copr = np.array([x_copr, y_copr, z_copr])
    Bh_traj = surfinBH._utils.transformTimeDependentVector(quat_nrsur, \
        Bh_traj_copr, inverse=0)

    return Bh_traj

#-----------------------------------------------------------------------------
def get_uniform_in_orbits_times(t, phi_orb, PTS_PER_ORBIT):
    """
    returns sparse time array such that there are PTS_PER_ORBIT points
    in each orbit.
    """
    # get numer of orbits
    n_orbits = int(abs((phi_orb[-1] - phi_orb[0])/(2*np.pi)))

    # get sparse times such that there are PTS_PER_ORBIT points in each orbit
    n_pts = int(n_orbits*PTS_PER_ORBIT)
    phi_orb_sparse = np.linspace(phi_orb[0], phi_orb[-1], n_pts)
    t_sparse = np.interp(phi_orb_sparse, phi_orb, t)

    return t_sparse

#----------------------------------------------------------------------------
def get_omegaOrb_from_sparse_data(t_sparse, phiOrb_sparse):
    """ Computes orbital frequency from sparse data using splines.
    """
    # spline interpolant for phase
    phiOrb_spl = UnivariateSpline(t_sparse, phiOrb_sparse, s=0)

    # spline for phase derivative
    omegaOrb_spl = phiOrb_spl.derivative()

    return omegaOrb_spl(t_sparse)

#----------------------------------------------------------------------------
def get_separation_from_omega(omega, mA, mB, chiA, chiB, LHat, pnorder=3.5):
    """ Roughly 3.5 PN accurate separation. This is not verified or tested,
    so don't use this for real science, only visualization. """

    eta = mA*mB
    deltaM = mA - mB

    Sigma_vec = mB*chiB - mA*chiA
    S_vec = mA**2.*chiA + mB**2.*chiB

    # some dot products
    chiAL = np.sum(LHat*chiA, axis=1)
    chiBL = np.sum(LHat*chiB, axis=1)
    chiAB = np.sum(chiA*chiB, axis=1)
    SigmaL = np.sum(Sigma_vec*LHat, axis=1)
    SL = np.sum(S_vec*LHat, axis=1)

    # Get 3.5 PN accurate gamma=1./r from Eq.(4.3) of
    # https://arxiv.org/pdf/1212.5520v2.pdf, but ignore the
    # log term in x**3 term

    x = omega**(2./3.)
    gamma_by_x = 0

    if pnorder >= 0:
        gamma_by_x += 1
    if pnorder >= 1:
        gamma_by_x += x * (1. - 1./3 *eta)
    if pnorder >= 1.5:
        gamma_by_x += x**(3./2) * (5./3 * SL + deltaM * SigmaL )
    if pnorder >= 2:
        gamma_by_x += x**2 * (1 - 65./12 *eta)
    if pnorder >= 2.5:
        gamma_by_x += x**(5./2) * ( (10./3 + 8./9 * eta)*SL \
            + 2* deltaM * SigmaL)
    if pnorder >= 3:
        gamma_by_x += x**3 * (1. + (-2203./2520 -41./192 * np.pi**2)*eta \
                + 229./36 * eta**2 + 1./81 * eta**3)
    if pnorder >= 3.5:
        gamma_by_x += x**(7./2) * ( (5 - 127./12 *eta - 6 * eta**2)*SL + \
                deltaM * SigmaL * (3 - 61./6 *eta - 8./3 * eta**2) )


    r = 1./gamma_by_x/x

    # To this add the 2PN spin-spin term from Eq.(4.13) of
    # https://arxiv.org/pdf/gr-qc/9506022.pdf
    if pnorder >= 2:
        r += omega**(-2./3) * (-1./2 * eta * chiAB) * omega**(4./3)

    return r

#----------------------------------------------------------------------------
def get_binary_data(q, chiA, chiB, omega_ref, PTS_PER_ORBIT, FREEZE_TIME, \
        omega_start=None, \
        uniform_time_step_size=None):

    mA = q/(1.+q)
    mB = 1./(1.+q)

    nr_sur = NRSur7dq2.NRSurrogate7dq2()

    # If omega_ref is not given, set f_ref to None, and t_ref to -100
    f_ref = None if omega_ref is None else omega_ref/np.pi
    t_ref = -100 if omega_ref is None else None

    # get NRSur dynamics
    quat_nrsur, orbphase_nrsur, _, _ = nr_sur.get_dynamics(q, chiA, chiB, \
        omega_ref=omega_ref, t_ref=t_ref, allow_extrapolation=True)

    if uniform_time_step_size is None:
        t_binary = get_uniform_in_orbits_times(nr_sur.tds, orbphase_nrsur, \
            PTS_PER_ORBIT)
    else:
        t_binary = np.arange(nr_sur.tds[0], nr_sur.tds[-1], \
            uniform_time_step_size)

    # If FREEZE_TIME is not in t_binary, add it
    if np.min(np.abs(t_binary - FREEZE_TIME)) > 0.1:
        t_binary = np.sort(np.append(t_binary, FREEZE_TIME))

    # If t=0 is not in t_binary, add it
    if np.min(np.abs(t_binary - 0)) > 0.1:
        t_binary = np.sort(np.append(t_binary, 0))

    # interpolate dynamics on to t_binary
    quat_nrsur = np.array([spline_interp(t_binary, nr_sur.tds, tmp) \
        for tmp in quat_nrsur])
    orbphase_nrsur = spline_interp(t_binary, nr_sur.tds, orbphase_nrsur)

    omega_nrsur = get_omegaOrb_from_sparse_data(t_binary, orbphase_nrsur)

    h_nrsur, chiA_nrsur, chiB_nrsur = nr_sur(q, chiA, chiB, \
        f_ref=f_ref, t_ref=t_ref, return_spins=True, \
        allow_extrapolation=True, t=t_binary)

    LHat = surfinBH._utils.lHat_from_quat(quat_nrsur).T
    separation = get_separation_from_omega(omega_nrsur, mA, mB, chiA_nrsur, \
        chiB_nrsur, LHat)

    # Newtonian
    LMag = q/(1.+q)**2 * omega_nrsur**(-1./3)
    L = LHat*LMag[:, None]

    # Get component trajectories
    BhA_traj = get_trajectory(separation * mB, quat_nrsur, orbphase_nrsur, 'A')
    BhB_traj = get_trajectory(separation * mA, quat_nrsur, orbphase_nrsur, 'B')

    # If omega_start is given, retain only higher frequencies
    if omega_start is not None:
        start_idx = np.argmin(np.abs(omega_nrsur - omega_start))

        t_binary = t_binary[start_idx:]
        chiA_nrsur = chiA_nrsur[start_idx:]
        chiB_nrsur = chiB_nrsur[start_idx:]
        L = L[start_idx:]
        BhA_traj = BhA_traj.T[start_idx:].T
        BhB_traj = BhB_traj.T[start_idx:].T
        separation = separation[start_idx:]
        for key in h_nrsur.keys():
            h_nrsur[key] = h_nrsur[key][start_idx:]

    return t_binary, chiA_nrsur, chiB_nrsur, L, h_nrsur, BhA_traj, \
        BhB_traj, separation

#----------------------------------------------------------------------------
def make_zero_if_small(x):
    if abs(x) < 1e-3:
        return 0
    else:
        return x

class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, \
        argparse.RawDescriptionHelpFormatter):
    pass

#----------------------------------------------------------------------------
def get_BH_shape(mass, chi):
    """ Get an ellipsoid according Kerr-Schild horizon for a BH, with center
    at origin, and spin along z-direction.
    """
    a_Kerr = mass*np.linalg.norm(chi)
    rplus = mass + np.sqrt(mass**2 - a_Kerr**2)
    equitorial_rad = np.sqrt(rplus**2 + a_Kerr**2)
    polar_rad = rplus

    # ellipsoid at origin
    u, v = np.meshgrid(np.linspace(0,2*np.pi,30), \
        np.arccos(np.linspace(-1,1,15)))
    x = equitorial_rad*np.cos(u)*np.sin(v)
    y = equitorial_rad*np.sin(u)*np.sin(v)
    z = polar_rad*np.cos(v)

    return [x, y, z]

#----------------------------------------------------------------------------
def black_hole_surface(shape_Bh, center, chi):
    """ Draws an ellipsoid according Kerr-Schild horizon for a BH.
        Takes an ellipsoid at origin, rotates polar axis along spin direction,
        and shifts center to BH center
    """

    x, y, z = np.copy(shape_Bh)
    array_shape = x.shape

    # rotate ellipsoid such that poles are along spin direction
    if np.linalg.norm(chi) > 1e-6:   # can't do norms, but don't need rotation

        # flatten meshgrid
        coords = np.vstack([x.ravel(), y.ravel(), z.ravel()])

        # rotate ellipsoid
        quat = quat_between_vecs([0,0,1], chi)
        coords = surfinBH._utils.transformTimeDependentVector(quat, coords, \
            inverse=0)

        # unflatten to get meshgrid again
        x = coords[0].reshape(array_shape)
        y = coords[1].reshape(array_shape)
        z = coords[2].reshape(array_shape)

    # shift center
    x += center[0]
    y += center[1]
    z += center[2]

    return x, y, z
