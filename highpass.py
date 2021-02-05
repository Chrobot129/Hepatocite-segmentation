#%%

from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_gradient_magnitude
import scipy.ndimage.morphology as mpg
from skimage.filters.thresholding import threshold_otsu, threshold_sauvola
import skimage.morphology as morph
import skimage.exposure as expo
from medpy import filter as mpflt
import skimage.segmentation as segment
import skimage.feature as ftr
from skimage.feature import blob_dog, blob_log, blob_doh
import skimage.filters as ftl
from math import sqrt
import skimage.color as clr
#%%

images = np.zeros((960,1280,3,17))

for i in range(17):
    images[:,:,:,i] = plt.imread("r{}.png".format(i+1))[:,:,0:3]


#%%
ball = morph.ball(radius = 5)

lowpass = np.zeros((960,1280,3,17))
gauss_highpass = np.zeros((960,1280,3,17))
images_trans = np.zeros((960,1280,3,17))
images_temp = np.zeros((960,1280,3,17))
images_bin = np.zeros((960,1280,3,17))

for i in range(17):

    images_temp[:,:,:,i] = mpflt.smoothing.anisotropic_diffusion(images[:,:,:,i], niter=10, option = 3, kappa = 100, gamma = 0.2,)
    lowpass[:,:,:,i] = ndimage.gaussian_filter(images_temp[:,:,:,i], 10)
    gauss_highpass[:,:,:,i] = images[:,:,:,i] - lowpass[:,:,:,i]
    gauss_highpass[:,:,:,i] = gauss_highpass[:,:,:,i]/np.max(np.abs(gauss_highpass[:,:,:,i]))

    images_trans[:,:,:,i] = ftl.laplace(gauss_highpass[:,:,:,i])
    images_trans[:,:,:,i] = mpg.morphological_gradient(images_trans[:,:,:,i], structure=ball)
    images_trans[:,:,:,i] = images_trans[:,:,:,i]/np.max(np.abs(images_trans[:,:,:,i]))
    images_trans[:,:,:,i] = expo.equalize_adapthist(images_trans[:,:,:,i])

    thresh = threshold_otsu(images_trans[:,:,:,i])
    images_bin[:,:,:,i] =  images_trans[:,:,:,i] > thresh

# %%
SMALL_SIZE = 30
MEDIUM_SIZE = 35
BIGGER_SIZE = 40

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)


fig, ax = plt.subplots(2,2, figsize = (40,30))

img_to_print = 15

ax[0,0].imshow(images[:,:,:,img_to_print])
ax[0,0].set_title("Oryginał")
ax[0,1].imshow(gauss_highpass[:,:,:,img_to_print])
ax[0,1].set_title("Filtracja Górnoprzepustowa")
ax[1,0].imshow(images_trans[:,:,:,img_to_print])
ax[1,0].set_title("Po wyrównaniu histogramu")
ax[1,1].imshow(images_bin[:,:,:,img_to_print])
ax[1,1].set_title("Wynik")
# %%

images_gray = np.zeros((960,1280,17))

i = 10

images_gray[:,:,i] = clr.rgb2gray(images_bin[:,:,:,i])
blobs_log = blob_log(images_gray[:,:,i], max_sigma=30, num_sigma=10, threshold=.1)
# Compute radii in the 3rd column.
blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
blobs_dog = blob_dog(images_gray[:,:,i], max_sigma=30, threshold=.1)
blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)
blobs_doh = blob_doh(images_gray[:,:,i], max_sigma=30, threshold=.01)



# %%
blobs_list = [blobs_log, blobs_dog, blobs_doh]
colors = ['yellow', 'lime', 'red']
titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
          'Determinant of Hessian']
sequence = zip(blobs_list, colors, titles)

fig, axes = plt.subplots(1, 3, figsize=(60, 20), sharex=True, sharey=True)
ax = axes.ravel()

for idx, (blobs, color, title) in enumerate(sequence):
    ax[idx].set_title(title)
    ax[idx].imshow(images_gray[:,:,i])
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
        ax[idx].add_patch(c)
    ax[idx].set_axis_off()

plt.tight_layout()
plt.show()
# %%
