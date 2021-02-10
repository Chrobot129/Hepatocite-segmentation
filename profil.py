#%%

from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.morphology as mpg
from skimage.filters.thresholding import threshold_otsu
import skimage.morphology as morph
import skimage.exposure as expo
from medpy import filter as mpflt
from skimage.feature import blob_dog, blob_log, blob_doh
import skimage.filters as ftl
from math import sqrt
import skimage.color as clr
import skimage.measure as msr

#%%

images = np.zeros((960,1280,3,17))

for i in range(17):
    images[:,:,:,i] = plt.imread("r{}.png".format(i+1))[:,:,0:3]

# %%

start = [0,0]
dst = [1000, 1279]
dst_rev = [1279, 1000]

img_to_print = 3

prof = msr.profile_line(images[:,:,:,img_to_print], src = start, dst= dst)
x = np.arange(prof.shape[0])

point1 = start
point2 = dst_rev

x_values = [point1[0], point2[0]]
y_values = [point1[1], point2[1]]

img_diffused = mpflt.smoothing.anisotropic_diffusion(images[:,:,:,img_to_print], niter=7, option = 1, kappa = 90, gamma = 0.2)
prof_diffusion = msr.profile_line(img_diffused, src = start, dst= dst)

median = np.median(prof_diffusion)

median_full = np.full(prof.shape[0], median)

fig, ax = plt.subplots(2,2, figsize = (40,40))

ax[0,0].plot(x_values, y_values, color="red") 
ax[0,0].imshow(images[:,:,:,img_to_print])
ax[0,0].set_axis_off()

ax[0,1].plot(x, prof[:,1])
ax[0,1].plot(x, prof[:,2])
ax[0,1].plot(x, prof[:,0])

ax[1,0].plot(x_values, y_values, color="red") 
ax[1,0].imshow(img_diffused)
ax[1,0].set_axis_off()

ax[1,1].plot(x, prof_diffusion[:,1])
ax[1,1].plot(x, prof_diffusion[:,2])
ax[1,1].plot(x, prof_diffusion[:,0])
ax[1,1].plot(x, median_full)


plt.tight_layout()
plt.show()

# %%
