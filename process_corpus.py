import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
from scipy import ndimage
from preprocess import *
from joblib import Parallel, delayed

CORPUS_DIR = '/data2/Kaggle/LungCan/stage1/'
TARGET_DIR = '/data2/Kaggle/LungCan/stage1_processed/sp1_morphseg/train/'

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

def get_image_e2e(slices):
    image = get_pixels_hu(slices)
    image, new_spacing = resample(image, slices, [1,1,1])
    segmented_mask_fill = segment_lung_mask(image, True)
    mask = ndimage.binary_dilation(segmented_mask_fill, iterations=1)
    image = image*mask
    image = zero_center(normalize(image))
    #image.shape+=(1,)
    return image

def convert_patient_dcm(pid, data_dir=CORPUS_DIR, target_dir=TARGET_DIR):
    '''Generator that yields pixel_array from dataset, and
    additionally the ID of the corresponding patient.'''
    
    slices = load_scan(data_dir+pid)
    img = get_image_e2e(slices).astype('float32')
    filename =  target_dir+pid+'.npy'
    np.save(filename, img)
    print(pid, 'processed')
    return 0

def check_corpus(filename='stage1_labels.csv'):
    df = pd.DataFrame.from_csv(filename)
    c = 0
    for pid in df.index:
    	pid = str(pid)
    	if not os.path.exists(CORPUS_DIR+pid):
    		print(pid, 'not found')
    	else:
    		c+=1
    print(c)
    return c

if __name__=='__main__':
	#check_corpus()
	#get_corpus_metadata()
	#for patient_dir in os.listdir(CORPUS_DIR):
	#convert_patient_dcm('0015ceb851d7251b8f399e39779d1e7d')
	df = pd.DataFrame.from_csv('stage1_labels.csv')
	PIDL = df.index.tolist()
	Parallel(n_jobs=16)(delayed(convert_patient_dcm)(pid) for pid in PIDL)
