import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os

def get_corpus_metadata(path='/data2/Kaggle/LungCan/stage1/'):
	print('id, nslices, shape, max, min ')
	for subpath in os.listdir(path):
		files = os.listdir(path+'/'+subpath)
		n_slices = len(files)
		ds = dicom.read_file(path+'/'+subpath+'/'+files[65])
		pixels = ds.pixel_array
		maxx, minn = np.amax(pixels), np.amin(pixels)
		shape = pixels.shape

		print(subpath, n_slices, shape, maxx, minn)
if __name__=='__main__':
	get_corpus_metadata()