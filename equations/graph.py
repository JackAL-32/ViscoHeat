import numpy as __np
import matplotlib.pyplot as __plt
from matplotlib.colors import Normalize, Colormap
from matplotlib import cm
from equations.parameters import *
import time
import os
import cv2

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
    vmin = 0

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

def graph_temp_gr(values, r, tha, shift=21, name=str(time.time())[7:-4], dir="./pictures/"):
    print(r.shape)
    print(tha.shape)
    print(values.shape)

    r = __np.concatenate((__np.flip(r[1:], axis=0), r), axis=0)
    
    R, theta = __np.meshgrid(r, tha)

    print(r.shape)
    print(tha.shape)
    print(values.shape)

    X = R * __np.cos(theta) * 1000
    Y = R * __np.sin(theta) * 1000

    fig, ax = __plt.subplots()
    ax.contourf(X, Y, values.T - shift, 256)  # Transpose values to match X, Y dimensions

    vmax = __np.ceil(__np.amax(__np.abs(values))) - shift
    vmin = 0

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


def create_video(data, dt, output_video='output.mp4', output_dir='frames/', frame_rate=20):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
       
    data = data - T0

    max_temp_across_frames = __np.max(data)
    
    sm = cm.ScalarMappable(cmap='jet', norm=Normalize(0, max_temp_across_frames))
   
    for i in range(data.shape[2]):
        frame_name = f"{output_dir}frame_{i:04d}.png"
        time_elapsed = i * dt
        graph_frame(data[:,:,i], sm, max_temp_across_frames, time_elapsed, frame_name)

    frames = [f"{output_dir}frame_{i:04d}.png" for i in range(len(os.listdir(output_dir)))]

    frame = cv2.imread(frames[0])
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

    for frame_path in frames:
        frame = cv2.imread(frame_path)
        out.write(frame)

    out.release()

def graph_frame(values, sm, max_temp, time_elapsed, filename):
    fig, ax = __plt.subplots()
    ax.contourf(X, Y, values, 256, cmap=sm.cmap, norm=Normalize(vmin=0, vmax=max_temp))
   
    tickstep = max_temp / 4
   
    cbar = __plt.colorbar(sm, ax=ax)
    cbar.set_ticks(__np.arange(0, max_temp + tickstep, tickstep))
   
    max_temp_index = __np.unravel_index(__np.argmax(values), values.shape)
    max_temp_value = values[max_temp_index]
    cbar.ax.axhline(max_temp_value, color='white', linewidth=2)
   
    ax.set_aspect("equal")
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("x (mm)")
    ax.set_title("Temperature (Degrees C Above Ambient)")
    
    ax.text(0.5, 0.95, f"Time: {round(time_elapsed, 4)} s", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='Black')

    __plt.savefig(filename)
    __plt.close()