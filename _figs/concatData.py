__author__ = "Melissa A. Klocke"
__email__ = "klocke@ucr.edu"
__version__ = "1.0"


import sys
import os
import pandas as pd
from glob import glob
import os
import re


def main():
	split_char = '_'
	files = glob('./*_data.csv')

	basdir, basename = os.path.split(files[0])
	fname, fext = os.path.splitext(basename)
	splitname = fname.split(split_char)
	fname_short = splitname[:-2]
	fname_short = split_char.join(fname_short)

	df = pd.DataFrame()

	for file in files:
		temp = pd.read_csv(file, index_col=0)
		df = pd.concat([df, temp], sort=False)

	df = df.reset_index(drop=True)

	df.to_csv('%s_allMeasurements.csv' % fname_short)





if __name__ == '__main__':
    main()