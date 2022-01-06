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
from skimage import (color, filters, measure, morphology, exposure)
from skimage.util import img_as_ubyte, img_as_uint


'''This script measures the area, approximate diameter, eccentricity, and number of fluorescently labeled blobs in a
Nikon nd2 epifluorescence micrograph. The approach is to separate the objects from the backgroun by an edge-based
threshold. This limits the measurements of objects outside the plane of focus, as those objects cannot be accurately measured.
This version doesn not separate nearby or touching objects. Please also see "edge_detectCondensates.py" for a version of this 
script with watershed segmentation, which is most appropriate for objects which are all a similar size. All user-input parameters 
are located in the main function and saved to csv for reproducibility and reference. There are additional lines of code throughout 
the functions to create more diagnostic images which can be included or commented out as desired. Further description of each 
function can be found below.

Call the script with "python edgeDetect_noSegmentation.py $nd2_filename" where $nd2_filename is your file. 
Call the script on 8 random images in a folder with "bash runDetectRandom8files_condensates.sh". 
Must have both scripts in the same directory as the .nd2 files.'''

def main():
	if len(sys.argv)==1:
		print("Need to specify file to process as a parameter.")
		print("   Exiting")
		exit()

	fn = sys.argv[1] 

	dil_s = 5 ## used for binary dilation of cleaned, thresholded image. Choose to make threshold results closest to majority of objects in samples
	sigma_val = 2.5 ## recommend 0.5-4
	thresh = 'otsu' ## or 'yen' or 'tri' or 'otsu', must be string matching those three choices exactly
	
	os.system("mkdir _figs")
	print("\nOpening file: ", fn)

	fname, img, pix_micron = nd2_read(fn)

	dilate_selem = morphology.disk(dil_s) 
	thresh_img = threshold(img, sigma_val, thresh, dilate_selem, fname)
	labeledImg = measure.label(thresh_img)
	
	prop_df = getCondensateMeasurements(labeledImg, img, pix_micron)
	label_img(img, labeledImg, prop_df, fname)
	prop_df.to_csv('_figs/%s_data.csv' % fname)

	dilate_s = 'morphology.disk(%s)' % dil_s
	saveRunValues(fn, fname, pix_micron, dilate_s, sigma_val, thresh)

	
def nd2_read(fn):
	'''Reads in the .nd2 file associated with the filename called with the script. 
	Returns the filename without the extension, the conversion from pixels to microns, and the image as a numpy array.'''
	basdir, basename = os.path.split(fn)
	fname, fext = os.path.splitext(basename)

	img = ND2Reader(fn)
	pix_micron = img.metadata['pixel_microns']
	img = np.array(img[0])
	return fname, img, pix_micron


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


def getCondensateMeasurements(labeledImg, img, pix_micron):
	'''Gets the below meeasurements for each labeled object in the image. Stores the information
	in a dataframe.'''
	props = measure.regionprops(labeledImg,img)

	areas = [r.area for r in props]
	meanIntensities = [r.mean_intensity for r in props]
	eqDiam = [r.equivalent_diameter for r in props]
	eccentricity = [r.eccentricity for r in props]
	label = [r.label for r in props]
	bbox = [r.bbox for r in props]

	eqDiam = [i * pix_micron for i in eqDiam]

	props_dict = {'Label':label, 'Areas (pix^2)': areas, 'Mean Intensity':meanIntensities, 
	'Equivalent Diameter (micron)':eqDiam, 'Eccentricity': eccentricity, 'BBox coords (pix)':bbox}
	df = pd.DataFrame(props_dict)
	return df


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


def saveRunValues(fn, fname, pix_m, dilate_selem, sigma, thresh):
	'''Saves the user-input parameters to a csv file for reproducibility and diagnostic purposes.'''
	dict_vals = {'filename': [fn], 'pix to micron': [pix_m], 'dilation selem': [dilate_selem], 'thresh sigma': [sigma], 
		'threshold type': [thresh]} 
	temp_df = pd.DataFrame(dict_vals)
	temp_df.to_csv('_figs/%s_runvalues.csv' % fname)










if __name__ == '__main__':
	main()