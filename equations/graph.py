import numpy as __np
import matplotlib.pyplot as __plt
from matplotlib.colors import Normalize, Colormap
from matplotlib import cm
from equations.parameters import *
import time
import matplotlib.animation as animation
import os

__plt.jet()

R, theta = __np.meshgrid(r, tha1)

X = R * __np.cos(theta) * 1000
Y = R * __np.sin(theta) * 1000

cmap = __plt.get_cmap('jet')

def graph_eq(sig, name=str(time.time())[7:-4], factor=10e6, dir="./pictures/"):
    fig, ax = __plt.subplots()
    values = __np.abs(sig)
    ax.contourf(X, Y, values, 256)

    vmax = __np.ceil(__np.amax(values) / factor)
    vmin = 0  # Set vmin to 0 for starting at 0

    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    tickstep = vmax / 4  # Calculate step for 5 ticks starting from 0

    fig.colorbar(sm, ticks=__np.arange(0, vmax + tickstep, tickstep), ax=ax)
    __plt.gca().set_aspect("equal")
    __plt.xlabel("z (mm)")
    __plt.ylabel("x (mm)")
    __plt.title("Heat " + name)
    fig.savefig(dir + name + ".png")

def graph_temp_gr(values, shift=21, name=str(time.time())[7:-4], dir="./pictures/"):
    fig, ax = __plt.subplots()
    ax.contourf(X, Y, values - shift, 256)

    vmax = __np.ceil(__np.amax(__np.abs(values))) - shift
    vmin = 0  # Set vmin to 0 for starting at 0

    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    tickstep = vmax / 4  # Calculate step for 5 ticks starting from 0

    fig.colorbar(sm, ticks=__np.arange(0, vmax + tickstep, tickstep), ax=ax)

    __plt.gca().set_aspect("equal")
    __plt.xlabel("z (mm)")
    __plt.ylabel("x (mm)")
    __plt.title(name)
    fig.savefig(dir + name + ".png")

# Define a function to generate frames and save them as images
def generate_frames(data, output_dir='frames/'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    max_temp = __np.max(data)
    num_frames = data.shape[2]

    for i in range(num_frames):
        fig_name = f"{output_dir}frame_{i:03d}.png"
        graph_frames(data[:,:,i], max_temp, fig_name)

# Define the function to plot temperature with graph_temp_gr
def graph_frames(values, max_temp, filename):
    fig, ax = __plt.subplots()
    contour = ax.contourf(X, Y, values, 256)

    vmin = 0
    vmax = max_temp

    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap='jet', norm=norm)
    sm.set_array([])

    tickstep = 10

    cbar = fig.colorbar(sm, ticks=__np.arange(0, vmax + tickstep, tickstep), ax=ax)
    
    # Find the index of the maximum temperature value
    max_temp_index = __np.unravel_index(__np.argmax(values), values.shape)
    max_temp_value = values[max_temp_index]
    
    # Add a marker at the maximum temperature value on the color bar
    cbar.ax.plot([0, 1], [max_temp_value, max_temp_value], color='white', linewidth=2)

    __plt.gca().set_aspect("equal")
    __plt.xlabel("z (mm)")
    __plt.ylabel("x (mm)")
    __plt.title("Temperature")
    __plt.savefig(filename)
    __plt.close()

# Define the function to create a video from the frames
def create_video(frames_dir='frames/', output_video='output.mp4'):
    frames = [f"{frames_dir}frame_{i:03d}.png" for i in range(len(os.listdir(frames_dir)))]
    frame_rate = 20  # Adjust the frame rate as needed

    # Create the video using ffmpeg
    os.system(f"ffmpeg -r {frame_rate} -i {frames_dir}frame_%03d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p {output_video}")
