import numpy as np
import xarray as xr
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

path = 'C:/Users/Guus van Hemert/Desktop/TW/TW5/Thesis/Data'

ctl = xr.open_dataset(path + '/CTL_AOD550.nc')
#ctl variables: lon, lat, time

ctl_masked = xr.open_dataset(path + '/CTL_AOD550_MASKED.nc')

nat = xr.open_dataset(path + '/NAT1_AOD550.nc')

spex_one = xr.open_dataset(path + '/SPEXone_Mask.nc')

index = []
for i in range(spex_one.Count.shape[0]):
    if not(np.isnan(spex_one.Count.data[i,:,:]).any()):
        index.append(i)

for i in range(ctl_masked.TAU_2D_550nm.shape[0]):
    if not(np.isnan(spex_one.Count.data[i,:,:]).any()):
        index.append(i)
        
for i in range(spex_one.Count.shape[1]):
    for j in range(spex_one.Count.shape[2]):
        if not(np.isnan(spex_one.Count.data[0,i,j])):
            index.append((i,j))
        
# ctl_masked en spex_one alleen maar nan
    

#%%
n = 5
start = 200
stop = start + n
times = np.arange(start=start, stop=stop)

fig, axes = plt.subplots(figsize=(20,5), ncols = 5)
for i in range(times.size):
    axes[i].set_title(ctl.time.data[times[i]])
    im = axes[i].imshow(ctl_masked.TAU_2D_550nm.data[times[i]], animated=True)
    
plt.show()

fig, axes = plt.subplots(figsize=(20,5), ncols = 5)
for i in range(times.size):
    axes[i].set_title(spex_one.time.data[times[i]])
    im = axes[i].imshow(spex_one.Count.data[times[i]], animated=True)
    
plt.show()


#Animation of the data
fig = plt.figure()
im = plt.imshow(ctl_masked.TAU_2D_550nm.data[0])
plt.colorbar(im)


def updatefig(i):
    im.set_array(ctl.TAU_2D_550nm.data[i])
    return im

#ani = animation.FuncAnimation(fig, updatefig, frames=ctl.TAU_2D_550nm.shape[0])
#ani.save('CTL_550nm.avi', writer='ffmpeg')


fig = plt.figure()
im = plt.imshow(ctl_masked.TAU_2D_550nm.data[0])
plt.colorbar(im)


def updatefig(i):
    im.set_array(ctl_masked.TAU_2D_550nm.data[i])
    return im

ani = animation.FuncAnimation(fig, updatefig, frames=ctl_masked.TAU_2D_550nm.shape[0])
ani.save(path + '/CTL_MASK_550nm.avi', writer='ffmpeg')

#%%
n_days = int(ctl.time.data.shape[0] / 8)
shape = (n_days, ctl.lat.shape[0], ctl.lon.shape[0])

ctl_day_avg = np.zeros(shape)
ctl_day_lat_grad = np.zeros(shape)
ctl_day_lon_grad = np.zeros(shape)

for n in range(n_days):
    ctl_day_avg[n,...] = np.mean(ctl.TAU_2D_550nm.data[n*8:(n+1)*8], axis=0)
    ctl_day_avg[n,...] = (ctl_day_avg[n,...] - np.min(ctl_day_avg)) / (
        np.max(ctl_day_avg[n,...]) - np.min(ctl_day_avg[n,...]))
    ctl_day_lat_grad[n,...] = np.gradient(ctl_day_avg[n,...], axis=0)
    ctl_day_lon_grad[n,...] = np.gradient(ctl_day_avg[n,...], axis=1)

    
#%% 

temp = spex_one
    
n_days = int(temp.time.data.shape[0] / 8)
shape = (n_days, temp.lat.shape[0], temp.lon.shape[0])

day_avg = np.zeros(shape)

for n in range(n_days):
    day_avg[n,...] = np.nanmean(temp.Count.data[n*8:(n+1)*8], axis=0)
    
fig = plt.figure()
im = plt.imshow(day_avg[0])
plt.colorbar(im)

def updatefig(i):
    im.set_array(day_avg[i])
    return im

ani = animation.FuncAnimation(fig, updatefig, frames=day_avg.shape[0])
ani.save(path + '/SPEX_ONE_DAY_AVG.avi', writer='ffmpeg')



    


