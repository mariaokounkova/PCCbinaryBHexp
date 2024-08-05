#!/usr/bin/env python

# HELLO I'm on the branch! my new branch!

# Imports
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
import matplotlib.colors as colorsgw_
from matplotlib.colors import SymLogNorm, Normalize
from matplotlib.colors import LinearSegmentedColormap

# Import some bbh explorer methods
from BBH_Auxiliary import *

# number of frames per orbit
PTS_PER_ORBIT = 30

# Time at which to freeze video for 5 seconds
FREEZE_TIME = -100

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
def get_spherical_grid(radius, num_theta, num_phi):
    theta = np.linspace(0, 2 * np.pi, num_theta)
    phi = np.linspace(0, np.pi, num_phi)

    # Create a meshgrid for theta and phi
    theta, phi = np.meshgrid(theta, phi)

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    return x, y, z, theta, phi

#----------------------------------------------------------------------------
def get_waveform_on_spherical_grid(t_vals, t_idx, h_dict, theta, phi, radius):
    """ Compute absolute value of strain at each r, th, ph value, using
    the retarded time.
    """
    h = np.zeros(theta.shape, dtype=complex)
    # find the time index that's closest to t_ret = t-r
    t = t_vals[t_idx]
    t_ret_idx = np.argmin(np.abs(t_vals - t + radius))
    for key in [(2, 2)]: #h_dict.keys():
        print("the key:", key)
        ell, m = key
        ylm = np.vectorize(harmonics.sYlm)(-2, ell, m, theta, phi)
        h += h_dict[key][t_ret_idx]*ylm
    return np.real(h/radius)

class AnimationWrapper:
    def __init__(self, fig, ax, t, \
        mA, BhA_traj, chiA_nrsur, \
        mB, BhB_traj, chiB_nrsur, \
        mf, BhC_traj, chif, \
        h_nrsur, max_range):

        self.fig = fig
        self.ax = ax
        self.t = t

        # Black hole properties
        self.mA = mA
        self.chiA_nrsur = chiA_nrsur
        self.BhA_traj = BhA_traj

        self.mB = mB
        self.chiB_nrsur = chiB_nrsur
        self.BhB_traj = BhB_traj

        self.mf = mf
        self.chif = chif
        self.BhC_traj = BhC_traj

        self.shape_BhA = get_BH_shape(mA, chiA_nrsur)
        self.shape_BhB = get_BH_shape(mB, chiB_nrsur)
        self.shape_BhC = get_BH_shape(mf, chif)
        
        # Black hole surfaces
        self.BhA = None
        self.BhB = None
        self.BhC = None

        self.max_range = max_range

        # Gravitational wave visualization
        self.h_nrsur = h_nrsur
        self.x, self.y, self.z, self.theta, self.phi = get_spherical_grid(max_range, 25, 25)
        self.gw = None

        # Time visualization
        self.time_text = ax.text2D(0.8, 0.8, '', transform=ax.transAxes, \
            fontsize=12, zorder=2, color='gray')

    def update(self, frame):

        # Update the time text
        current_time = self.t[frame]
        self.time_text.set_text(f'$t={current_time:.1f}$')

        #linthresh = 0.0001
        vmax = 0.03
        vmin = -vmax
        norm = Normalize(vmin=vmin, vmax=vmax)

        # Plot gravitiational wave
        h_spherical = get_waveform_on_spherical_grid(self.t, frame - 1, \
                    self.h_nrsur, self.theta, self.phi, self.max_range)
        if self.gw is not None:
                 self.gw.remove()
        self.gw = self.ax.plot_surface(self.x, self.y, self.z, \
            facecolors=cm.coolwarm(norm(h_spherical)), \
            alpha = 0.2, linewidth=0, antialiased=False)

        print("frame: %d, time: %.1f, max h: %.4f, min h: %.4f, vmax: %.4f, vmin: %.4f" \
           % (frame, current_time, np.max(h_spherical), np.min(h_spherical), vmax, vmin))
        
        # Plot black holes before merger
        if current_time < 0:
            # Remove the black hole surfaces for redrawing
            if self.BhA is not None:
                self.BhA.remove()
            if self.BhB is not None:
                self.BhB.remove()
 
            # Draw the black holes
            X, Y, Z = black_hole_surface(self.shape_BhA, self.BhA_traj[:,frame-1], \
                      self.chiA_nrsur[frame-1])
            self.BhA = self.ax.plot_surface(X, Y, Z, color='#2f1170', linewidth=0, alpha=0.5, zorder=1)
 
            X, Y, Z = black_hole_surface(self.shape_BhB, self.BhB_traj[:,frame-1], \
                      self.chiB_nrsur[frame-1])
            self.BhB = self.ax.plot_surface(X, Y, Z, color='black', linewidth=0, alpha=0.5, zorder=1)

        # Plot black hole after merger
        else: 
            ## Clear binary stuff
            if abs(current_time) < 10:
                if self.BhA is not None:
                    self.BhA.remove()
                if self.BhB is not None:
                    self.BhB.remove()
                self.BhA = None
                self.BhB = None

            if self.BhC is not None:
                self.BhC.remove()

            # Draw the black hole
            X, Y, Z = black_hole_surface(self.shape_BhC, self.BhC_traj[:,frame-1], self.chif)
            self.BhC = self.ax.plot_surface(X, Y, Z, color='k', linewidth=0, alpha=0.5, zorder=1)

#----------------------------------------------------------------------------
def BBH_animation(q, chiA, chiB, save_file):

    #######################
    ## Getting Physics Data
    #######################

    chiA = np.array(chiA)
    chiB = np.array(chiB)

    mA = q/(1.+q)
    mB = 1./(1.+q)

    # evaluate remnant fit
    fit = surfinBH.LoadFits('surfinBH7dq2')

    omega_ref = None
    t_binary, chiA_nrsur, chiB_nrsur, L, h_nrsur, BhA_traj, \
         BhB_traj, separation = get_binary_data(q, chiA, chiB, omega_ref, \
         PTS_PER_ORBIT, FREEZE_TIME, \
         omega_start = None, uniform_time_step_size = None)

    # Outer radius of visualization
    max_range = 1.1*np.nanmax(np.linalg.norm(BhB_traj, axis=0))
    print("max_range", max_range)

    # Time parameters
    waveform_end_time = 50 + 2*max_range
    dt_remnant = 100

    # Common time and frames
    t = np.append(t_binary[t_binary < waveform_end_time], \
        np.arange(waveform_end_time, 500 + waveform_end_time, dt_remnant))
    frames = [1, 2, 3] #range(1, 300) #range(len(t))

    # Remnant properties
    mf, chif, vf, mf_err, chif_err, vf_err \
         = fit.all(q, chiA, chiB, omega0=omega_ref)
    BhC_traj = np.array([v * t for v in vf])

    #######################
    ## Plotting
    #######################

    # Create figure and 3D axis
    fig = plt.figure()

    # Background image
    #background_image = plt.imread('cosmos.png')
    #ax_img = fig.add_axes([0, 0, 1, 1], zorder=-1)  # Full screen
    #ax_img.imshow(background_image, aspect='auto', alpha = 0.9)
    #ax_img.axis('off')  # Hide the 2D axis

    ax = fig.add_subplot(111, projection='3d')

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

    # Set aspect ratio to be 1:1:1
    ax.set_box_aspect([1,1,1])

    # Create an instance of the wrapper class
    anim_wrapper = AnimationWrapper(fig, ax, t, \
            mA, BhA_traj, chiA_nrsur, \
            mB, BhB_traj, chiB_nrsur, \
            mf, BhC_traj, chif, \
            h_nrsur, max_range)

    # Create the animation object
    ani = animation.FuncAnimation(fig, anim_wrapper.update, frames=frames, blit=False)

    # Save the animation
    print("Saving the animation")
    writer = animation.FFMpegWriter(fps=20, metadata=dict(artist='Maria Okounkova'), bitrate=1800)
    ani.save(save_file, writer=writer)

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

