import fnmatch
import os
import re
import threading
import dicom
import pandas
import numpy as np
import tensorflow as tf
import skimage
import skimage.io
import skimage.transform
from preprocess import *
#G
def get_corpus_size(directory, pattern='*.dcm'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return len(files)

def find_files(directory, pattern='*.dcm'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files

def get_image_e2e(slices):
    image = get_pixels_hu(slices)
    image, new_spacing = resample(image, slices, [1,1,1])
    segmented_mask_fill = segment_lung_mask(image, True)
    image = image*segmented_mask_fill
    image = zero_center(normalize(image))
    return image

def load_individual_dcm(directory):
    '''Generator that yields pixel_array from dataset, and
    additionally the ID of the corresponding patient.'''
    files = find_files(directory)
    for filename in files:
        ds = dicom.read_file(filename)
        img = get_image_e2e([ds])
        yield img, str(ds.PatientID)

def load_patient_npy(path):
    if not path.endswith('/'): 
        path += '/'
    for fname in os.listdir(path):
        img = np.load(path+fname)
        pid = fname.split('.')[0]
        yield img, pid

def load_patient(path):
    '''Generator that yields pixel_array from dataset, and
    additionally the ID of the corresponding patient.'''
    if not path.endswith('/'): 
        path += '/'
    for fname in os.listdir(path):
        if fname.endswith('.npy'):
            img = np.load(path+fname)
            pid = fname.split('.')[0]
        else:
            slices = load_scan(path+fname)
            img = get_image_e2e(slices)
            pid = fname
        img.shape+=(1,)
        yield img, pid

def load_label_df(filename='stage1_labels.csv'):
    df = pandas.DataFrame.from_csv(filename)
    return df


class DCMReader(object):
    '''Generic background reader that preprocesses files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 data_dir,
                 coord,
                 e2e=False,    #TODO: not yet implemented, for preprocessed inputs
                 threshold=None,
                 queue_size=16, 
                 byPatient=True,
                 q_shape=None,
                 pattern='*.npy'):
        self.data_dir = data_dir
        self.coord = coord
        self.e2e = e2e
        self.threshold = threshold
        self.ftype = pattern
        self.corpus_size = get_corpus_size(self.data_dir, pattern=self.ftype)
        self.threads = []
        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.label_placeholder = tf.placeholder(dtype=tf.int32, shape=[], name='label') #!!!
        if q_shape:
            self.queue = tf.FIFOQueue(queue_size,['float32'], shapes=q_shape)
        else:
            self.q_shape = [(None, None, None, 1)] if byPatient else [(None, None, 1)]
            self.queue = tf.PaddingFIFOQueue(queue_size,
                                             ['float32'],
                                             shapes=self.q_shape)
        self.enqueue = self.queue.enqueue([self.sample_placeholder])
        self.queue_l = tf.FIFOQueue(queue_size,
                                         'int32',
                                         shapes=[])
        self.enqueue_l = self.queue_l.enqueue([self.label_placeholder])
        self.byPatient = byPatient
        self.labels_df = load_label_df()

    def dequeue(self, num_elements):
        output = self.queue.dequeue_many(num_elements)
        labels = self.queue_l.dequeue_many(num_elements)
        return output, labels

    def thread_main(self, sess):
        buffer_ = np.array([])
        stop = False
        # Go through the dataset multiple times
        while not stop:
            if self.byPatient:
                iterator = load_patient(self.data_dir)
            else:
                iterator = load_individual_dcm(self.data_dir)
            for img, patient_id in iterator:
                #print(filename)
                try: 
                    label = self.labels_df['cancer'][patient_id]
                except(KeyError):
                    print('No match for ', patient_id)
                    continue
                if self.coord.should_stop():
                    stop = True
                    break
                if self.threshold is not None:
                    #TODO:  Perform quality check if needed
                    pass

                else:
                    sess.run(self.enqueue,
                             feed_dict={self.sample_placeholder: img})
                    sess.run(self.enqueue_l,
                             feed_dict={self.label_placeholder: label})

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads
