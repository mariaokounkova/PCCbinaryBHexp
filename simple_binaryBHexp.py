#!/usr/bin/env python

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
from matplotlib.colors import SymLogNorm
from matplotlib.colors import LinearSegmentedColormap

# number of frames per orbit
PTS_PER_ORBIT = 30

# Time at which to freeze video for 5 seconds
FREEZE_TIME = -100

colors_dict = {
        'BhA_traj': 'white',
        'BhB_traj': 'white'
        }

zorder_dict = {
        'contourf': -200,
        'info_text': 200,
        'notice_text': 100,
        'traj': 100,
        'Bh': 150,
        }

colors = [
    (0, "#ff99ff"),    # Pastel pink
    (0.25, "#4de1ff"), # Light blue
    (0.5, "black"),    # Black at the midpoint (value zero)
    (0.75, "#ffcc66"), # Light orange
    (1, "#ff6666")     # Pastel red
]

# Create the custom colormap
gw_cmap = LinearSegmentedColormap.from_list("custom_coolwarm_black", colors)


#----------------------------------------------------------------------------
def quat_between_vecs(u, v):
    """ Computes quaternion for rotation from one vector to another
    """

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
def draw_black_hole(shape_Bh, center, chi):
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
def get_grids_on_planes(num_pts_1d, max_range):
    # generate grid
    x_1d = np.linspace(-max_range, max_range, num_pts_1d)
    y_1d = np.linspace(-max_range, max_range, num_pts_1d)
    z_1d = np.linspace(-max_range, max_range, num_pts_1d)

    [xZ, yZ] = np.meshgrid(x_1d, y_1d)
    [xY, zY] = np.meshgrid(x_1d, z_1d)
    [yX, zX] = np.meshgrid(y_1d, z_1d)

    xX = zZ = -max_range
    yY = max_range

    # Get Euclidean radii and th,ph
    rZ = np.sqrt(xZ**2 + yZ**2 + zZ**2)
    thZ = np.arccos(zZ/rZ)
    phZ = np.arctan2(yZ,xZ)

    rY = np.sqrt(xY**2 + yY**2 + zY**2)
    thY = np.arccos(zY/rY)
    phY = np.arctan2(yY,xY)

    rX = np.sqrt(xX**2 + yX**2 + zX**2)
    thX = np.arccos(zX/rX)
    phX = np.arctan2(yX,xX)

    return [rX, thX, phX], [yX, zX], \
           [rY, thY, phY], [xY, zY], \
           [rZ, thZ, phZ], [xZ, yZ]

#----------------------------------------------------------------------------
def get_waveform_on_grid(t_vals, t_idx, h_dict, sph_grid):
    """ Compute absolute value of strain at each r, th, ph value, using
    the retarded time.
    """
    r, th, ph = sph_grid
    h = np.zeros(r.shape, dtype=complex)
    # find the time index that's closest to t_ret = t-r
    t = t_vals[t_idx]
    t_ret_idx = np.vectorize(lambda r: np.argmin(np.abs(t_vals - t + r)))(r)
    for key in h_dict.keys():
        ell, m = key
        ylm = np.vectorize(harmonics.sYlm)(-2, ell, m, th, ph)
        h += h_dict[key][t_ret_idx]*ylm
    return np.real(h/r)

#----------------------------------------------------------------------------
def get_waveform_timeseries(h_dict, azim, elev):
    """ Compute the timeseries to plot in the lower panel from a given viewpoint
    """
    ph = azim * np.pi/180.
    th = (90. - elev) * np.pi/180.
    h = np.zeros_like(h_dict[list(h_dict.keys())[0]], dtype=complex)
    for key in h_dict.keys():
        ell, m = key
        ylm = harmonics.sYlm(-2, ell, m, th, ph)
        h += h_dict[key]*ylm
    return h

#----------------------------------------------------------------------------
def get_binary_data(q, chiA, chiB, omega_ref, omega_start=None, \
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

class AnimationWrapper:
    def __init__(self, fig, ax, lines, t, dataLines_binary, dataLines_remnant, \
        mA, shape_BhA, chiA_nrsur, \
        mB, shape_BhB, chiB_nrsur, \
        mf, shape_BhC, BhC_traj, chif, 
        sph_gridZ, gridZ, h_nrsur, max_range):
        self.fig = fig
        self.ax = ax
        self.lines = lines
        self.t = t
        self.dataLines_binary = dataLines_binary
        self.dataLines_remnant = dataLines_remnant
        self.mA = mA,
        self.shape_BhA = shape_BhA
        self.chiA_nrsur = chiA_nrsur
        self.BhA = None
        self.mB = mB,
        self.shape_BhB = shape_BhB
        self.chiB_nrsur = chiB_nrsur
        self.BhB = None
        self.mf = mf
        self.shape_BhC = shape_BhC
        self.BhC_traj = BhC_traj
        self.chif = chif
        self.BhC = None
        self.sph_gridZ = sph_gridZ
        self.gridZ = gridZ
        self.h_nrsur = h_nrsur
        self.max_range = max_range
        self.time_text = ax.text2D(0.8, 0.8, '', transform=ax.transAxes, fontsize=12, zorder=zorder_dict['info_text'], color='white')

    def update(self, frame):

        # Update the time text
        current_time = self.t[frame]
        self.time_text.set_text(f'$t={current_time:.1f}$')
        print("time: %.1f" % current_time)

        hplusZ = get_waveform_on_grid(self.t, frame-1, self.h_nrsur, self.sph_gridZ)
        # color range for contourf
        # Get linthresh from first index. With SymLogNorm, whenever the
        # value is less than linthresh, the color scale is linear. Else log.
        linthresh = 0.1 #np.max(np.abs(get_waveform_on_grid(self.t, 0, self.h_nrsur, self.sph_gridZ)))
        # Get vmax from waveform at peak.  Add in propagation delay
        zero_idx = np.argmin(np.abs(self.t - self.max_range))
        vmax = np.max(get_waveform_on_grid(self.t, zero_idx, self.h_nrsur, \
                                           self.sph_gridZ))
        # Symmetric about 0
        vmin = -vmax
        linthresh = abs(vmax)/100
        norm = SymLogNorm(linthresh=linthresh, linscale=1, vmin=vmin, vmax=vmax)
        self.ax.contourf(self.gridZ[0], self.gridZ[1], hplusZ, zdir='z', \
                offset=-self.max_range, cmap=cm.coolwarm, \
                zorder=zorder_dict['contourf'], vmin=vmin, vmax=vmax, norm=norm)

        ## Plot black holes before merger
        if current_time < 0:
            # Remove the black hole surfaces for redrawing
            if self.BhA is not None:
                self.BhA.remove()
            if self.BhB is not None:
                self.BhB.remove()
            # Update the lines

            for idx, line in enumerate(self.lines):
                data = self.dataLines_binary[idx]
                line.set_data(data[0:2, :frame])
                line.set_3d_properties(data[2, :frame])

            # Draw the black holes
            X, Y, Z = draw_black_hole(self.shape_BhA, self.dataLines_binary[0][:,frame-1], self.chiA_nrsur[frame-1])
            self.BhA = self.ax.plot_surface(X, Y, Z, color='k', linewidth=4, alpha=0.5, zorder=zorder_dict['Bh'])
 
            X, Y, Z = draw_black_hole(self.shape_BhB, self.dataLines_binary[1][:,frame-1], self.chiB_nrsur[frame-1])
            self.BhB = self.ax.plot_surface(X, Y, Z, color='k', linewidth=0, alpha=0.1, zorder=zorder_dict['Bh'])

        ## Plot black hole after merger
        else: 
            ## Clear binary stuff
            if abs(current_time) < 10:
                if self.BhA is not None:
                    self.BhA.remove()
                if self.BhB is not None:
                    self.BhB.remove()
                self.BhA = None
                self.BhB = None
                for idx, line in enumerate(self.lines):
                    line.set_data(np.array([]), np.array([]))
                    line.set_3d_properties(np.array([]))

            if self.BhC is not None:
                self.BhC.remove()

            # Draw the black hole
            X, Y, Z = draw_black_hole(self.shape_BhC, self.BhC_traj[:,frame-1], self.chif)
            self.BhC = self.ax.plot_surface(X, Y, Z, color='k', linewidth=0, alpha=0.1, zorder=zorder_dict['Bh'])


        return self.lines + [self.BhA] + [self.BhB] + [self.BhC]


#----------------------------------------------------------------------------
def BBH_animation(q, chiA, chiB, save_file, omega_ref=None, \
        draw_full_trajectory=False, project_on_all_planes=False, \
        height_map=False, auto_rotate_camera=False, \
        still_time=None,  rescale_fig_for_widgets=False, \
        no_freeze_near_merger=False, omega_start=None, \
        no_wave_time_series=False, uniform_time_step_size=None, \
        no_time_label=False, no_surrogate_label=False, \
        use_spin_angular_momentum_for_arrows=False):

    #######################
    ## Getting Physics Data
    #######################

    chiA = np.array(chiA)
    chiB = np.array(chiB)
    t_binary, chiA_nrsur, chiB_nrsur, L, h_nrsur, BhA_traj, \
         BhB_traj, separation = get_binary_data(q, chiA, chiB, omega_ref, \
         omega_start=omega_start, uniform_time_step_size=uniform_time_step_size)
    max_range = 1.1*np.nanmax(np.linalg.norm(BhB_traj, axis=0))
    print("max_range", max_range)
    mA = q/(1.+q)
    mB = 1./(1.+q)

    sph_gridX, gridX, sph_gridY, gridY, sph_gridZ, gridZ \
        = get_grids_on_planes(20, max_range)

    # evaluate remnant fit
    fit_name = 'surfinBH7dq2'
    fit = surfinBH.LoadFits(fit_name)

    # If omega_ref is None, will assume the spins are given in the
    # # coorbital frame at t=-100M
    mf, chif, vf, mf_err, chif_err, vf_err \
         = fit.all(q, chiA, chiB, omega0=omega_ref)

    # Get Bh shapes assuming fixed spin magnitudes
    shape_BhA = get_BH_shape(mA, chiA)
    shape_BhB = get_BH_shape(mB, chiB)
    shape_BhC = get_BH_shape(mf, chif)

    #######################
    ## Plotting
    #######################

    # Plotting properties
    # Create figure and 3D axis
    fig = plt.figure()

    # Background image
    background_image = plt.imread('cosmos.png')
    ax_img = fig.add_axes([0, 0, 1, 1], zorder=-1)  # Full screen
    ax_img.imshow(background_image, aspect='auto', alpha = 0.9)
    ax_img.axis('off')  # Hide the 2D axis

    ax = fig.add_subplot(111, projection='3d')

    properties_fontsize = 10
    properties_text_yloc = 0.8
    freeze_fontsize = 14
    timestep_fontsize = 12

    ax.set_xlim3d([-max_range*0.96, max_range*0.96])
    ax.set_ylim3d([-max_range*0.96, max_range*0.96])
    ax.set_zlim3d([-max_range*0.96, max_range*0.96])

    ax.grid(False)  # Remove grid lines

    # Fully remove the panes
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)

    # Optionally remove the pane lines
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')

    # Set the background color of the 3D plot to be transparent
    ax.patch.set_alpha(0)

    # Time parameters
    waveform_end_time = 50 + 2*max_range
    dt_remnant = 100
    hist_frames = int(0.75*(PTS_PER_ORBIT))

    # common time array: After waveform_end_time, each step is 100M
    t = np.append(t_binary[t_binary < waveform_end_time], \
        np.arange(waveform_end_time, 500 + waveform_end_time, dt_remnant))
    frames = range(2) #range(len(t))

    # assume merger is at origin
    BhC_traj = np.array([tmp*t for tmp in vf])

    dataLines_binary = [BhA_traj, BhB_traj, 1, 1, 1]
    dataLines_remnant = [1]

    ## Plotting elements
    trajectory_A_line = ax.plot(BhA_traj[0,0:1]-1e10, BhA_traj[1,0:1], BhA_traj[2,0:1], \
            color=colors_dict['BhA_traj'], lw = 2, alpha = 0.5, \
            zorder=zorder_dict['traj'])[0]

    trajectory_B_line = ax.plot(BhB_traj[0,0:1]-1e10, BhB_traj[1,0:1], BhB_traj[2,0:1], \
            color=colors_dict['BhB_traj'], lw = 2, alpha = 0.5, \
            zorder=zorder_dict['traj'])[0]

    lines = [trajectory_A_line, trajectory_B_line]

    # Create an instance of the wrapper class
    anim_wrapper = AnimationWrapper(fig, ax, lines, t, dataLines_binary, dataLines_remnant, \
            mA, shape_BhA, chiA_nrsur, \
            mB, shape_BhB, chiB_nrsur, \
            mf, shape_BhC, BhC_traj, chif,
            sph_gridZ, gridZ, h_nrsur, max_range)

    # Create the animation object
    ani = animation.FuncAnimation(fig, anim_wrapper.update, frames=frames, blit=False)

    # Save the animation
    print("Saving the animation")
    writer = animation.FFMpegWriter(fps=20, metadata=dict(artist='Maria Okounkova'), bitrate=1800)
    ani.save(save_file, writer=writer)

class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, \
        argparse.RawDescriptionHelpFormatter):
    pass

#############################    main    ##################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=CustomFormatter)

    pp_standard = parser.add_argument_group("Standard options")

    pp_standard.add_argument('--q', type=float, required=True,
        help='Mass ratio. Currently 1 <= q <= 2.')
    pp_standard.add_argument('--chiA', type=float, required=True, nargs=3,
        help='Dimensionless spin of BhA at omega_ref. List of size 3.')
    pp_standard.add_argument('--chiB', type=float, required=True, nargs=3,
        help='Dimensionless spin of BhB at omega_ref. List of size 3.')
    pp_standard.add_argument('--save_file', type=str, required=True, 
        help='File to save animation to. If given, will save animation to ' \
            'this file. Else will show animation. Use this option if live ' \
            'rendering is slow. Allowed extensions are mp4 and gif. mp4 has ' \
            'the best quality. We use lower quality for gif to reduce file ' \
            'size. Example: --save_file movie.mp4')

    args = parser.parse_args()
    BBH_animation(args.q, args.chiA, args.chiB, args.save_file)

    #plt.show()

