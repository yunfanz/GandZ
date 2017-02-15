import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
from scipy import ndimage
from preprocess import *
from joblib import Parallel, delayed

SP2_BOX = (210, 180, 210)
#CORPUS_DIR = '/data2/Kaggle/LungCan/stage1/'
#TARGET_DIR = '/data2/Kaggle/LungCan/stage1_processed/sp2_noseg/train/'
MASK_DIR = '/data2/Kaggle/LungCan/stage1_processed/sp2_waterseg/masks/'
CORPUS_DIR = '/home/yunfanz/Data/Kaggle/LungCan/stage1/'
TARGET_DIR = '/home/yunfanz/Data/Kaggle/LungCan/stage1_processed/sp1_morphseg/masks/'
#MASK_DIR = '/home/yunfanz/Data/Kaggle/LungCan/stage1_processed/sp1_morphseg/masks/'
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

def get_image_e2e(slices, return_mask=False):
    image = get_pixels_hu(slices)
    image, new_spacing = resample(image, slices, [1,1,1])
    segmented_mask_fill = segment_lung_mask(image, True)
    mask = ndimage.binary_dilation(segmented_mask_fill, iterations=1)
    image = image*mask
    image = zero_center(normalize(image))
    #image.shape+=(1,)
    if return_mask: 
        return image, mask
    else:
        return image

def get_masks(pid, data_dir=CORPUS_DIR, target_dir=MASK_DIR):
    '''Generator that yields pixel_array from dataset, and
    additionally the ID of the corresponding patient.'''
    
    slices = load_scan(data_dir+pid)
    mask = get_image_e2e(slices, return_mask=True)[1].astype('int8')
    filename =  target_dir+'mask_'+pid+'.npy'
    np.save(filename, mask)
    print(pid, 'processed')
    return 0

def convert_patient_dcm(pid, data_dir=CORPUS_DIR, target_dir=TARGET_DIR, mask_dir=None, bbox=False):
    '''Generator that yields pixel_array from dataset, and
    additionally the ID of the corresponding patient.'''
    
    slices = load_scan(data_dir+pid)
    img, mask = get_image_e2e(slices, return_mask=True)
    img = img.astype('float32')
    mask = mask.astype('int8')

    if bbox:
        zmin, zmax, rmin, rmax, cmin, cmax = bbox_3d(mask)
        img = img[zmin:zmax,rmin:rmax,cmin:cmax]
        #mask = mask[zmin:zmax,rmin:rmax,cmin:cmax]
    filename =  target_dir+pid+'.npy'
    
    if mask_dir:
        mfname = mask_dir+'mask_'+pid+'.npy'
        np.save(mfname, mask)
    np.save(filename, img)
    print(pid, 'processed')
    return 0

def test_convert(pid, data_dir=CORPUS_DIR, target_dir=TARGET_DIR, mask_dir=None, bbox=False, segment=False):
    '''Generator that yields pixel_array from dataset, and
    additionally the ID of the corresponding patient.'''
    #print('targetdir', target_dir)
    #print('mask_dir', mask_dir)
    assert os.path.exists(target_dir) 

    # if os.path.exists(target_dir+pid+'.npy'):
    #     print('file', pid, 'exists, skipping.')
    #     return
    if not os.path.exists(data_dir+pid):
        print('no', pid)
        return
    slices = load_scan(data_dir+pid)
    image = get_pixels_hu(slices)
    image, new_spacing = resample(image, slices, [2,2,2])
    if segment:
        assert os.path.exists(mask_dir)
        try:
            mask = np.vstack([np.expand_dims(watershed_seg_2d(zslice,mode='f_only'),axis=0) for zslice in image])
            #mask = watershed_seg_3d(image, mode='f_only')
        except:
            print('ERROR Processing', pid)
            return
        image = image*mask
        mask = mask.astype('int8')
        if mask_dir:
            mfname = mask_dir+'mask_'+pid+'.npy'
            np.save(mfname, mask)
        #segmented_mask_fill = segment_lung_mask(image, True)
        #mask, _ = resample(water_seg, slices, [2,2,2])
        # img, _ = resample(img, slices, [2,2,2])
        #mask = ndimage.binary_dilation(water_seg, iterations=1)
        #img = img*mask
    image = zero_center(normalize(image))
    image = image.astype('float32')
    
    
    if bbox:
        zmin, zmax, rmin, rmax, cmin, cmax = bbox_3d(mask)
        image = image[zmin:zmax,rmin:rmax,cmin:cmax]
        #mask = mask[zmin:zmax,rmin:rmax,cmin:cmax]
    filename =  target_dir+pid+'.npy'
    
    
    np.save(filename, image)
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

def get_box_sizes(path=MASK_DIR, mode='pandas', verbose=False):
    if mode == 'pandas':
        boxes = pd.DataFrame(columns=['zmin','zmax','rmin','rmax','cmin','cmax'])
        i=0
        for file in os.listdir(path):
            mask = np.load(path+file)
            zmin, zmax, rmin, rmax, cmin, cmax = bbox_3d(mask)
            pid = file.split('.')[0].split('_')[1]
            boxes.loc[pid] = {'zmin':zmin, 'zmax':zmax, 'rmin':rmin, 'rmax':rmax, 'cmin':cmin, 'cmax':cmax}
            if verbose: print(i, pid)
            i+=1
    elif mode=='numpy':
        boxes = []
        for file in os.listdir(path):
            mask = np.load(path+file)
            zmin, zmax, rmin, rmax, cmin, cmax = bbox_3d(mask)
            boxes.append([zmin, zmax, rmin, rmax, cmin, cmax])
        boxes = np.array(boxes)
    return boxes

def apply_bbox(data_dir, target_dir, mask_dir=MASK_DIR, pid=None, sizes=SP2_BOX):
    if pid:
        if pid.endswith('.npy'):
            pid = pid.split('.')[0]
        try:
            img = np.load(data_dir+pid+'.npy')
            mask = np.load(mask_dir+'mask_'+pid+'.npy')
        except:
            print(pid, 'does not exist')
            return
        hs = [s//2 for s in sizes]
        val = img[0,0,0]
        #print('padding with', val)
        img = np.pad(img,((hs[0], hs[0]),(hs[1],hs[1]), (hs[2],hs[2])), mode='constant', constant_values=val )
        mask = np.pad(mask,((hs[0], hs[0]),(hs[1],hs[1]), (hs[2],hs[2])), mode='constant', constant_values=0 )
        center2,_ = corner_to_center_and_size(*bbox_3d(mask))
        z1,r1,c1 = (center2[0]-sizes[0])//2,(center2[1]-sizes[1])//2,(center2[2]-sizes[2])//2
        z2,r2,c2 = (center2[0]+sizes[0])//2,(center2[1]+sizes[1])//2,(center2[2]+sizes[2])//2
        img = img[z1:z2, r1:r2, c1:c2]
        np.save(target_dir+pid+'.npy', img)
        return
    else:    
        for fname in os.listdir(data_dir):
            pid = fname.split('.')[0]
            apply_bbox(data_dir,mask_dir,target_dir,pid=pid, sizes=sizes)



if __name__=='__main__':
    #check_corpus()
    #get_corpus_metadata()
    #for patient_dir in os.listdir(CORPUS_DIR):
    #convert_patient_dcm('0015ceb851d7251b8f399e39779d1e7d')

    df = pd.DataFrame.from_csv('stage1_labels.csv')
    #df = pd.DataFrame.from_csv('stage1_sample_submission.csv')
    PIDL = df.index.tolist()
    Parallel(n_jobs=4)(delayed(test_convert)(pid) for pid in PIDL)


    # to_dir = '/home/yunfanz/Projects/Kaggle/LungCan/DATA/train/'
    # from_dir = '/data2/Kaggle/LungCan/stage1_processed/sp2_waterseg/train/'
    # PIDL = os.listdir(from_dir)
    #Parallel(n_jobs=12)(delayed(apply_bbox)(from_dir, to_dir, pid=pid) for pid in PIDL)


