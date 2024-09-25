#!/usr/bin/env python

# Imports
import numpy as np
import os
import argparse
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import InterpolatedUnivariateSpline
import surfinBH
import NRSur7dq2
from NRSur7dq2 import harmonics
import pyvista as pv

# Import some bbh explorer methods
from BBH_Auxiliary import *

#----------------------------------------------------------------------------
def get_waveform_on_spherical_grid(t_vals, t_idx, h_dict, sphere):
    """ Compute absolute value of strain at t - r. """
    h = np.zeros(sphere.theta.shape, dtype = complex)
    # find the time index that's closest to t-r
    t = t_vals[t_idx]
    t_ret_idx = np.argmin(np.abs(t_vals - t + sphere.radius))

    ## Visualizing the (2,2) mode for now! 
    for key in [(2, 2)]: #h_dict.keys():
        ell, m = key
        ylm = np.vectorize(harmonics.sYlm)(-2, ell, m, sphere.theta, sphere.phi)
        h += h_dict[key][t_ret_idx]*ylm

    return np.real(h / sphere.radius)

class Sphere:
    """ Create a spherical grid """
    def __init__(self, radius, num_theta, num_phi):
        self.radius = radius
        theta = np.linspace(0, 2 * np.pi, num_theta)
        phi = np.linspace(0, np.pi, num_phi)
        self.theta, self.phi = np.meshgrid(theta, phi)
        self.x = radius * np.sin(self.phi) * np.cos(self.theta)
        self.y = radius * np.sin(self.phi) * np.sin(self.theta)
        self.z = radius * np.cos(self.phi)


class AnimationWrapper:
    """ Class for animation data """
    def __init__(self, t, \
        mA, BhA_traj, chiA_nrsur, \
        mB, BhB_traj, chiB_nrsur, \
        mf, BhC_traj, chif, \
        h_nrsur, save_dir):

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

        # Gravitational wave visualization
        self.radii = [10, 20, 25, 30]
        self.spheres = [Sphere(rad, 15, 15) for rad in self.radii]
        self.h_nrsur = h_nrsur

        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Make x, y, and z values for the fixed points
        x_values = np.array([])
        y_values = np.array([])
        z_values = np.array([])
        for sphere in self.spheres:
            x_values = np.append(x_values, sphere.x.ravel())
            y_values = np.append(y_values, sphere.y.ravel())
            z_values = np.append(z_values, sphere.z.ravel())
        self.points = np.c_[x_values, y_values, z_values]

    def update(self, frame):
        """ Get the animation data for the current frame """

        # Update the time text
        current_time = self.t[frame]
        print("Generated data for frame %d" % frame)

        # Plot gravitiational wave
        h_values = np.array([])
        
        for sphere in self.spheres:
            h_spherical = get_waveform_on_spherical_grid(self.t, frame - 1, \
                    self.h_nrsur, sphere)
            h_values = np.append(h_values, h_spherical.ravel())

        values = h_values

        # TODO:
        # Here is where you take points and values and dump them in a data format appropriate
        # for your visualization software! 

        # For example: If you want to use Paraview!
        grid = pv.PolyData(self.points)
        grid["GWStrain"] = values
        filename = self.save_dir + f"/gw_strain_t{frame:03d}.vtk"
        grid.save(filename)
        print(f"Written: {filename}")

        # # TODO: Visualizing the black hole horizonts
        # # Before merger
        # if current_time < 0:
        
        #     # Black hole shapes
        #     Xa, Ya, Za = black_hole_surface(self.shape_BhA, self.BhA_traj[:,frame-1], \
        #               self.chiA_nrsur[frame-1])
 
        #     Xb, Yb, Zb = black_hole_surface(self.shape_BhB, self.BhB_traj[:,frame-1], \
        #               self.chiB_nrsur[frame-1])
        # # After merger
        # else: 
        #     # Final black hole shape
        #     Xc, Yc, Zc = black_hole_surface(self.shape_BhC, self.BhC_traj[:,frame-1], self.chif)

#----------------------------------------------------------------------------
def BBH_animation(q, chiA, chiB, save_dir):
    """ Wrapper to get physics data and call the animation! """

    chiA = np.array(chiA)
    chiB = np.array(chiB)

    mA = q / (1 + q)
    mB = 1 / (1 + q)

    # evaluate remnant fit
    fit = surfinBH.LoadFits('surfinBH7dq2')

    omega_ref = None
    pts_per_orbit = 30
    freeze_time = -100
    t_binary, chiA_nrsur, chiB_nrsur, L, h_nrsur, BhA_traj, \
         BhB_traj, separation = get_binary_data(q, chiA, chiB, omega_ref, \
         pts_per_orbit, freeze_time, \
         omega_start = None, uniform_time_step_size = None)

    max_range = 1.1 * np.nanmax(np.linalg.norm(BhB_traj, axis = 0))

    # Time parameters
    waveform_end_time = 50 + 2 * max_range
    dt_remnant = 100

    # Common time and frames
    t = np.append(t_binary[t_binary < waveform_end_time], \
        np.arange(waveform_end_time, 500 + waveform_end_time, dt_remnant))

    # Remnant properties
    mf, chif, vf, mf_err, chif_err, vf_err \
         = fit.all(q, chiA, chiB, omega0 = omega_ref)
    BhC_traj = np.array([v * t for v in vf])

    # Create an instance of the wrapper class
    anim_wrapper = AnimationWrapper(t, \
            mA, BhA_traj, chiA_nrsur, \
            mB, BhB_traj, chiB_nrsur, \
            mf, BhC_traj, chif, \
            h_nrsur, save_dir)

    # Frames you want to visualize (here I just put the first two frames)
    # TODO: Change to include the frames you want! Line range(len(t))
    frames = [1, 2] #range(len(t))

    # Perform the animation! 
    for frame in frames:
        anim_wrapper.update(frame)


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
    pp_standard.add_argument('--save_directory', type=str, required=True, 
        help='Directory to same the .vtk files to')

    args = parser.parse_args()
    BBH_animation(args.q, args.chiA, args.chiB, args.save_directory)

