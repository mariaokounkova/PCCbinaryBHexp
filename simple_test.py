import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os 

## TODO: change this to the path to your ffmpeg (or remove this argument)
os.environ['FFMPEG_EXE'] = 'virtualenv/bin/ffmpeg'

# Example update function
def update_lines(frame, line, x, y, z):
    z = np.sin(x + frame / 10.0) * np.cos(y + frame / 10.0)
    line.set_data(x, y)
    line.set_3d_properties(z)
    return line,

# Data for plotting
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
z = np.sin(x) * np.cos(y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initial plot
line, = ax.plot(x, y, z)

# Additional arguments for the update function
fargs = (line, x, y, z)

print("Creating the animation")
# Creating the animation
line_ani = FuncAnimation(fig, update_lines, frames=100, fargs=fargs,
                         interval=50, blit=False, repeat=True, repeat_delay=5000)

print("Saving the animation")
# Save the animation
writer = FFMpegWriter(fps=20, metadata=dict(artist='Maria Okounkova'), bitrate=1800)
line_ani.save("test_movie.mp4", writer=writer)

plt.close(fig)
