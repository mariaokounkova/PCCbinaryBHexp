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
    h = np.zeros(sphere.theta.shape, dtype=complex)
    # find the time index that's closest to t-r
    t = t_vals[t_idx]
    t_ret_idx = np.argmin(np.abs(t_vals - t + sphere.radius))

    ## Visualizing the (2,2) mode for now! 
    for key in [(2, 2)]:  # Change this to include other modes if needed
        ell, m = key
        ylm = np.vectorize(harmonics.sYlm)(-2, ell, m, sphere.theta, sphere.phi)
        h += h_dict[key][t_ret_idx] * ylm

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

        # Create faces for the spherical mesh
        self.faces = self.create_faces(num_theta, num_phi)

    def create_faces(self, num_theta, num_phi):
        """ Generate faces for the spherical grid """
        faces = []
        for i in range(num_theta - 1):
            for j in range(num_phi - 1):
                # Two triangles per square face
                faces.append([3, i * num_phi + j,
                              (i + 1) * num_phi + j,
                              (i + 1) * num_phi + (j + 1)])  # Triangle 1
                faces.append([3, i * num_phi + j,
                              (i + 1) * num_phi + (j + 1),
                              i * num_phi + (j + 1)])  # Triangle 2
        return np.array(faces)

class AnimationWrapper:
    """ Class for animation data """
    def __init__(self, t, mA, BhA_traj, chiA_nrsur, mB, BhB_traj, chiB_nrsur, mf, BhC_traj, chif, h_nrsur, save_dir):
        self.t = t
        self.mA = mA
        self.chiA_nrsur = chiA_nrsur
        self.BhA_traj = BhA_traj
        self.mB = mB
        self.chiB_nrsur = chiB_nrsur
        self.BhB_traj = BhB_traj
        self.mf = mf
        self.chif = chif
        self.BhC_traj = BhC_traj
        self.h_nrsur = h_nrsur

        self.radii = [10, 20, 25, 30]
        self.spheres = [Sphere(rad, 15, 15) for rad in self.radii]

        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def update(self, frame):
        """ Get the animation data for the current frame """
        current_time = self.t[frame]
        print("Generated data for frame %d" % frame)

        # Create PyVista plotter
        plotter = pv.Plotter()

        # Generate data for each sphere
        for sphere in self.spheres:
            # Get the h values using the get_waveform_on_spherical_grid function
            h_values = get_waveform_on_spherical_grid(self.t, frame, self.h_nrsur, sphere)

            # Create PyVista PolyData object for the sphere
            points = np.c_[sphere.x.ravel(), sphere.y.ravel(), sphere.z.ravel()]
            faces = sphere.faces
            
            mesh = pv.PolyData(points, faces.ravel())

            # Add scalar values to the mesh
            mesh["GWStrain"] = h_values.ravel()

            # Add the mesh to the plotter with lower opacity and colormap
            plotter.add_mesh(mesh, scalars="GWStrain", cmap="coolwarm", opacity=0.3, \
                show_scalar_bar=True, clim=[-0.001, 0.001])

        # Show the plotter
        plotter.show()

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
    t_binary, chiA_nrsur, chiB_nrsur, L, h_nrsur, BhA_traj, BhB_traj, separation = get_binary_data(q, chiA, chiB, omega_ref, pts_per_orbit, freeze_time)

    max_range = 1.1 * np.nanmax(np.linalg.norm(BhB_traj, axis=0))

    # Time parameters
    waveform_end_time = 50 + 2 * max_range
    dt_remnant = 100

    # Common time and frames
    t = np.append(t_binary[t_binary < waveform_end_time], np.arange(waveform_end_time, 500 + waveform_end_time, dt_remnant))

    # Remnant properties
    mf, chif, vf, mf_err, chif_err, vf_err = fit.all(q, chiA, chiB, omega0=omega_ref)
    BhC_traj = np.array([v * t for v in vf])

    # Create an instance of the wrapper class
    anim_wrapper = AnimationWrapper(t, mA, BhA_traj, chiA_nrsur, mB, BhB_traj, chiB_nrsur, mf, BhC_traj, chif, h_nrsur, save_dir)

    # Frames you want to visualize (here I just put the first two frames)
    frames = [1] #range(len(t))  # Change this to visualize the desired frames

    # Perform the animation! 
    for frame in frames:
        anim_wrapper.update(frame)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    pp_standard = parser.add_argument_group("Standard options")
    pp_standard.add_argument('--q', type=float, required=True, help='Mass ratio. Currently 1 <= q <= 2.')
    pp_standard.add_argument('--chiA', type=float, required=True, nargs=3, help='Dimensionless spin of BhA at omega_ref. List of size 3.')
    pp_standard.add_argument('--chiB', type=float, required=True, nargs=3, help='Dimensionless spin of BhB at omega_ref. List of size 3.')
    pp_standard.add_argument('--save_directory', type=str, required=True, help='Directory to save the output.')

    args = parser.parse_args()
    BBH_animation(args.q, args.chiA, args.chiB, args.save_directory)
