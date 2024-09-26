#!/usr/bin/env python

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

def get_waveform_on_spherical_grid(t_vals, t_idx, h_dict, sphere):
    """ Given a spherical grid with radius r and a time, compute 
        h(t - r) """
    h = np.zeros(sphere.theta.shape, dtype = complex)
    # find the time index that's closest to t - r
    t = t_vals[t_idx]
    t_ret_idx = np.argmin(np.abs(t_vals - t + sphere.radius))

    # Evaluate the desired gravitational wave modes
    for key in [(2, 2), (2, -2)]: # Change this to include other modes if needed
        ell, m = key
        ylm = np.vectorize(harmonics.sYlm)(-2, ell, m, sphere.theta, sphere.phi)
        h += h_dict[key][t_ret_idx] * ylm

    return np.real(h / sphere.radius)

class Sphere:
    """ Create a spherical grid given a radius and angular points """
    def __init__(self, radius, num_theta, num_phi):

        self.radius = radius
        theta = np.linspace(0, 2 * np.pi, num_theta)
        phi = np.linspace(0, np.pi, num_phi)
        self.theta, self.phi = np.meshgrid(theta, phi)
        self.x = radius * np.sin(self.phi) * np.cos(self.theta)
        self.y = radius * np.sin(self.phi) * np.sin(self.theta)
        self.z = radius * np.cos(self.phi)
        self.points = np.c_[self.x.ravel(), self.y.ravel(), self.z.ravel()]

        # Create faces for the spherical mesh
        self.faces = self.create_faces(num_theta, num_phi)
        self.mesh = pv.PolyData(self.points, self.faces.ravel())

    def create_faces(self, num_theta, num_phi):
        """ Generate faces for the spherical grid, created with ChatGPT """
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
    """ Wrapper for the BBH animation given black hole masses, spins, trajectories, 
        and gravitational wave data  """
    def __init__(self, t, mA, BhA_traj, chiA_nrsur, \
                          mB, BhB_traj, chiB_nrsur, \
                          mF, BhF_traj, chiF, h_nrsur, save_dir):
        self.t = t
        self.mA = mA
        self.chiA_nrsur = chiA_nrsur
        self.BhA_traj = BhA_traj
        self.mB = mB
        self.chiB_nrsur = chiB_nrsur
        self.BhB_traj = BhB_traj
        self.mF = mF
        self.chiF = chiF
        self.BhF_traj = BhF_traj
        self.h_nrsur = h_nrsur

        self.radii = [40, 80, 120, 160, 200]
        self.spheres = [Sphere(rad, 15, 15) for rad in self.radii]


        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def update(self, frame):
        """ Get the animation data for the current frame """

        current_time = self.t[frame]
        print("Generated data for time %d" % current_time)

        # Create PyVista plotter
        plotter = pv.Plotter(off_screen=True)
        plotter.clear()

        # Generate data for each sphere
        for sphere in self.spheres:

            # Get values for the waveform
            h_values = get_waveform_on_spherical_grid(self.t, frame, self.h_nrsur, sphere)

            # Add scalar values to the mesh
            mesh = sphere.mesh
            mesh["GWStrain"] = h_values.ravel()

            # Add the mesh to the plotter with lower opacity and colormap
            plotter.add_mesh(mesh, scalars = "GWStrain", cmap = "RdBu", opacity = 0.05,
                show_scalar_bar = False, clim = [-0.0005, 0.0005])

    
        # Set the camera position and focal point
        plotter.camera.position = 0.5*np.array(plotter.camera.position) # Position of the camera
        plotter.camera.roll = 45.0

        plotter.set_background("black")  # Set background color
        plotter.add_background_image("cosmos.jpg")

        # Render the plotter
        plotter.show(screenshot = True) 


        # Save a screenshot for the current frame
        screenshot_file = f"{self.save_dir}/frame_{frame:03d}.png"
        plotter.screenshot(screenshot_file)

        # Close the plotter to free resources
        #plotter.close()
        del plotter

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
    BhF_traj = np.array([v * t for v in vf])

    # Create an instance of the wrapper class
    anim_wrapper = AnimationWrapper(t, mA, BhA_traj, chiA_nrsur, \
        mB, BhB_traj, chiB_nrsur, \
        mf, BhF_traj, chif, h_nrsur, save_dir)

    # Frames you want to visualize 
    frames = range(5) #np.arange(1,100,5) #range(20) #range(len(t)) 

    # Perform the animation! 
    for frame in frames:
        anim_wrapper.update(frame)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    pp_standard = parser.add_argument_group("Standard options")
    pp_standard.add_argument('--q', type=float, \
        required=True, help='Mass ratio. Currently 1 <= q <= 2.')
    pp_standard.add_argument('--chiA', type=float, \
        required=True, nargs=3, help='Dimensionless spin of BhA at omega_ref. List of size 3.')
    pp_standard.add_argument('--chiB', type=float, \
        required=True, nargs=3, help='Dimensionless spin of BhB at omega_ref. List of size 3.')
    pp_standard.add_argument('--save_directory', type=str, required=True, help='Directory to save the output.')

    args = parser.parse_args()
    BBH_animation(args.q, args.chiA, args.chiB, args.save_directory)
