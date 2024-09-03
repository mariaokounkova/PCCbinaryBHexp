import numpy as np
from pyvtk import VtkData, PointData, Scalars, StructuredGrid

def create_spherical_wave_data(t, num_radial=20, num_theta=20, num_phi=20):
    # Define the spherical coordinate grid
    r = np.linspace(0.1, 5, num_radial)  # Radial distance from the origin
    theta = np.linspace(0, np.pi, num_theta)  # Polar angle from the z-axis
    phi = np.linspace(0, 2 * np.pi, num_phi)  # Azimuthal angle from the x-axis

    # Create 3D meshgrid for spherical coordinates
    R, Theta, Phi = np.meshgrid(r, theta, phi, indexing='ij')

    # Convert spherical coordinates to Cartesian coordinates
    X = R * np.sin(Theta) * np.cos(Phi)
    Y = R * np.sin(Theta) * np.sin(Phi)
    Z = R * np.cos(Theta)

    # Compute the wave amplitude in spherical coordinates
    wave_amplitude = np.sin(2 * np.pi * R - 2 * np.pi * t)

    # Reshape data for VTK format
    points = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    values = wave_amplitude.ravel()

    return points, values

def write_vtk_file(points, values, filename):
    # Create an UnstructuredGrid (more flexible for spherical data)
    grid = StructuredGrid(points.shape[0], points)

    # Create VTK data structure
    vtk_data = VtkData(
        grid,
        PointData(
            Scalars(values, name='WaveAmplitude')
        )
    )

    # Write to .vtk file
    vtk_data.tofile(filename)

if __name__ == "__main__":
    num_timesteps = 10
    for timestep in range(num_timesteps):
        t = timestep * 0.1  # Define the time step interval
        points, values = create_spherical_wave_data(t)
        filename = f"vtk_data/spherical_wave_t{timestep:03d}.vtk"
        write_vtk_file(points, values, filename)
        print(f"Written: {filename}")
