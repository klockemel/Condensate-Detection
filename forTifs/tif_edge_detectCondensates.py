__author__ = "Melissa A. Klocke"
__email__ = "klocke@ucr.edu"
__version__ = "1.0"


import sys
import os
import imageio as io
from nd2reader import ND2Reader
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import (color, filters, measure, morphology, segmentation, segmentation, exposure)
from skimage.util import img_as_ubyte, img_as_uint, invert
from skimage.feature import peak_local_max


'''This script measures the area, approximate diameter, eccentricity, and number of fluorescently labeled blobs in a
Nikon nd2 epifluorescence micrograph. The approach is to separate the objects from the backgroun by an edge-based
threshold. This limits the measurements of objects outside the plane of focus, as those objects cannot be accurately measured.
This version separates nearby or touching objects using a watershed algorithm, which is more reliable if the objects are all 
of a similar size. Please also see "edgeDetect_noSegmentation.py" for a version of this script without watershed segmentation.
All user-input parameters are located in the main function and saved to csv for reproducibility and reference. There are 
additional lines of code throughout the functions to create more diagnostic images which can be included or commented out as 
desired. Further description of each function can be found below.

Call the script with "python edge_detectCondensates.py $tif_filename" where $tif_filename is your file. 
Call the script on 8 random images in a folder with "bash runt_DetectRandom8files_condensates.sh". 
Must have both scripts in the same directory as the .tif files.'''


def main():
	if len(sys.argv)==1:
		print("Need to specify file to process as a parameter.")
		print("   Exiting")
		exit()

	fn = sys.argv[1] 

	dil_s = 5 ## used for binary dilation of cleaned, thresholded image. Choose to make threshold results closest to majority of objects in samples
	sigma_val = 2.5 ## recommend 0.5-4
	thresh = 'otsu' ## or 'yen' or 'tri' or 'otsu', must be string matching those three choices exactly
	footprint=np.ones((30, 30))	# used in watershed segmentation, adjusted based on the size of your objects of interest in pixels
	min_s = 20	## used in watershed segmentation, will help merge local minima 
	pix_micron = 0.070556640625 ## for 60x img, must be updated to your microscopes setting

	os.system("mkdir _figs")
	print("\nOpening file: ", fn)

	fname, img = tif_read(fn) 

	dilate_selem = morphology.disk(dil_s)
	thresh_img = threshold(img, sigma_val, thresh, dilate_selem, fname)

	minDilation_selem = morphology.disk(min_s)
	labeledImg, prop_df = watershed(img, thresh_img, sigma_val, footprint, minDilation_selem, pix_micron, fname)

	label_img(img, labeledImg, prop_df, fname)
	prop_df.to_csv('_figs/%s_data.csv' % fname)

	dilate_s = 'morphology.disk(%s)' % dil_s
	minDilation_s = 'morphology.disk(%s)' % min_s
	saveRunValues(fn, fname, pix_micron, dilate_s, sigma_val, thresh, footprint.shape, minDilation_s)


def tif_read(fn):
	'''Reads in the .tif file associated with the filename called with the script. 
	Returns the filename without the extension and the image as a numpy array.'''
	basdir, basename = os.path.split(fn)
	fname, fext = os.path.splitext(basename)
	img = img_as_uint(io.imread(fn))
	return fname, img


def threshold(img, sigma, thresh, dilate_selem, fn):
	'''Applies an edge-based threshold to the image. The ideal threshold for the sample should be
	decided by visual confirmation with the raw image. Once the image is thresholded, it is passed through
	some binary operations to clean it and increase the size of the objects (as the edge-based threshold
	otherwise tends to underestimate the size of objects).'''
	gauss_filtered_img = filters.gaussian(img, sigma=sigma)

	edges = filters.sobel(gauss_filtered_img)
	if thresh == 'otsu':
		thresh_val = filters.threshold_otsu(edges)
	elif thresh == 'yen':
		thresh_val = filters.threshold_yen(edges)
	elif thresh == 'tri':
		thresh_val = filters.threshold_triangle(edges)

	maskImg = edges > thresh_val

	thin = morphology.skeletonize(maskImg)
	filled = ndi.binary_fill_holes(thin)
	cleaned = morphology.binary_opening(filled)
	cleaned = morphology.binary_dilation(cleaned, selem=dilate_selem)

	## The following are optional diagnostic images
	# io.imwrite('_figs/%s_Edgethresh.tif' % fn, img_as_uint(maskImg), format = 'tif')
	# io.imwrite('_figs/%s_thinEdgethresh.tif' % fn, img_as_uint(thin), format = 'tif')
	# io.imwrite('_figs/%s_fillEdgethresh.png' % fn, img_as_uint(filled), format = 'png')
	# io.imwrite('_figs/%s_thresh.png' % fn, img_as_uint(cleaned), format = 'png')  
	return cleaned


def getCondensateMeasurements(labels_localmax_markers, img, pix_micron):
	'''Gets the below meeasurements for each labeled object in the image. Stores the information
	in a dataframe.'''
	props = measure.regionprops(labels_localmax_markers,img)

	areas = [r.area for r in props]
	meanIntensities = [r.mean_intensity for r in props]
	eqDiam = [r.equivalent_diameter for r in props]
	centroid = [r.centroid for r in props]
	eccentricity = [r.eccentricity for r in props]
	label = [r.label for r in props]
	bbox = [r.bbox for r in props]

	eqDiam = [i * pix_micron for i in eqDiam]

	props_dict = {'Label':label, 'Areas (pix^2)': areas, 'Mean Intensity':meanIntensities, 
	'Equivalent Diameter (micron)':eqDiam, 'Eccentricity': eccentricity, 'BBox coords (pix)':bbox}
	df = pd.DataFrame(props_dict)
	return df


def watershed(img, thresh, sigma, fp, selem, pix_micron, fn):
	'''Segments objects in the image which are near or touching. This function has been adapted from a method
	in Alon Oyler-Yaniv's DIP workshop notebooks: https://github.com/alonyan/DIP
	Both the "fp" and "selem" can be adjusted to improve segmentation based on the size of the objects in the 
	image. Most effective for objects which are relatively uniform in size. If objects vary widely in size, 
	consider using the "edgeDetect_noSegmentation.py" script which does not apply segmentation to avoid over- or
	under-segmentation.'''
	masked_image = img * thresh
	smoothed_masked_image = filters.gaussian(masked_image, sigma=sigma)
	inverted_smoothed_masked_image = invert(smoothed_masked_image)

	image_to_watershed = inverted_smoothed_masked_image

	imagePeaks = peak_local_max(-image_to_watershed, footprint=fp)
	MaskedImagePeaks = np.zeros_like(img, dtype=bool)
	MaskedImagePeaks[tuple(imagePeaks.T)] = True

	peakMask = morphology.dilation(MaskedImagePeaks, selem)
	markers = measure.label(peakMask)

	labels_localmax_markers = segmentation.watershed(image_to_watershed, markers, watershed_line=1, mask=thresh)
	df = getCondensateMeasurements(labels_localmax_markers, img, pix_micron)

	## The following are optional diagnostic images
	# io.imwrite('_figs/%s_thresh.png' % fn, final_img, format = 'png') 
	# plt.imsave('_figs/%s_watershed_edges.png' % fn, filters.sobel(labels_localmax_markers), cmap='nipy_spectral')
	# plt.imsave('_figs/%s_watershed.png' % fn, labels_localmax_markers, cmap='nipy_spectral')
	# io.imwrite('_figs/%s_watershed.png' % fn, labels_localmax_markers, format = 'png')
	# io.imwrite('_figs/%s_watershed_edges.png' % fn, filters.sobel(labels_localmax_markers), format = 'png')
	return labels_localmax_markers, df


def draw_blobs(img, labeledImg):
	'''Draws the outline of detected objects in red.'''
	vmin, vmax = np.percentile(img, q=(0.05, 99.95))
	img_equalized = exposure.rescale_intensity(img, in_range=(vmin, vmax), out_range=np.float32)
	img_equalized = exposure.equalize_adapthist(img_equalized)
	img = color.gray2rgb(img_as_ubyte(img_equalized))
	edges = filters.sobel(labeledImg) > 0.
	img[edges] = (220, 20, 20)
	return img


def label_img(img, labeledImg, df, fn):
	'''Creates a diagnostic image in which detected objects are labeled with a numerical ID on the original iamge. 
	The ID corresponds to the ID on the on output csv file with the measurements of each object. The original image 
	has histogram equalization applied to improve visibility.'''
	x_coord = [num[-1] for num in df['BBox coords (pix)']]
	y_coord = [num[0] for num in df['BBox coords (pix)']]
	label = df['Label'].tolist()

	img_blobs = draw_blobs(img, labeledImg)

	fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
	ax.imshow(img_blobs)
	for i in range(len(x_coord)):
		ax.text(x_coord[i],y_coord[i],str(label[i]),fontsize=15, color='palegoldenrod')

	ax.set_title('Detected condensates (red) on droplet image')
	fig.tight_layout()
	fig.savefig('_figs/%s_compimg.%s' % (fn.replace("/","__"), 'png'), dpi=700)
	plt.close()


def saveRunValues(fn, fname, pix_m, dilate_selem, sigma, thresh, fp, minDilation_selem):
	'''Saves the user-input parameters to a csv file for reproducibility and diagnostic purposes.'''
	dict_vals = {'filename': [fn], 'pix to micron': [pix_m], 'dilation selem': [dilate_selem], 'thresh, watershed sigma value': [sigma], 
		'threshold type': [thresh], 'watershed footprint': [fp], 'merging minimum dilation selem': [minDilation_selem]} 
	temp_df = pd.DataFrame(dict_vals)
	temp_df.to_csv('_figs/%s_runvalues.csv' % fname)










if __name__ == '__main__':
    main()