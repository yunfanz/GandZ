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
#G

class PatientImageProcesor(object):
    '''Image processor for a patient. Loads data from *.dcm files by default and processes.'''

	def __init__(self, patient_directory):
        self.files = self.find_files(patient_directory)
        self.corpus_size = len(self.files)

	def find_files(self,directory, pattern='*.dcm'):
        '''Recursively finds all files matching the pattern.'''
        files = []
        for root, dirnames, filenames in os.walk(directory):
            for filename in fnmatch.filter(filenames, pattern):
                files.append(os.path.join(root, filename))
        return files

    def load_generic_dcm(self, directory, resize=None):
        '''Generator that yields images from the directory.'''
        files = find_files(directory)
        for filename in files:
            ds = dicom.read_file(filename)
            img = ds.pixel_array
            audio = audio.reshape(-1, 1)
            yield img, filename

    def add_slice_thickness(self):
        '''Add the slice thickness to the meta data'''
        self.        


    def rescale_2_hu(self,dcm):
        '''Rescale image to hounsfield unit.'''
        dcm[dcm==-2000] = 0.0
        if dcm.RescaleSlope != 1:
            dcm.pixel_array = dcm.RescaleSlope*dcm.pixel_array.astype(np.float64)
            dcm.pixel_array = dcm.pixel_array.astype(np.int16)
        dcm.pixel_array += np.int16(dcm.RescaleIntercept)

        return dcm.pixel_array
        

    def load_patient_dcm(self, directory, resize=None):
        '''Generator that yields pixel_array from dataset, and
        additionally the ID of the corresponding patient.'''
        files = find_files(directory)
        for filename in files:
            ds = dicom.read_file(filename)
            img = ds.pixel_array
            img = img / 2000.0
            #assert (0 <= img).all() and (img <= 1.0).all() !!!
            if resize and (ds.pixel_array.shape != (resize, resize)):
                short_edge = min(img.shape[:2])
                yy = int((img.shape[0] - short_edge) / 2)
                xx = int((img.shape[1] - short_edge) / 2)
                crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
                # resize to 512, 512
                img = skimage.transform.resize(crop_img, (resize, resize))
            img = np.expand_dims(img, axis=-1)
            yield img, str(ds.PatientID)

    def load_label_df(filename='stage1_labels.csv'):
        df = pandas.DataFrame.from_csv(filename)
        return df


class DCMReader(object):
    '''Generic background reader that preprocesses files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 data_dir,
                 coord,
                 resize=None,
                 threshold=None,
                 queue_size=64):
        self.data_dir = data_dir
        self.coord = coord
        self.resize = resize
        self.threshold = threshold
        self.corpus_size = get_corpus_size(self.data_dir)
        self.threads = []
        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.label_placeholder = tf.placeholder(dtype=tf.int32, shape=[], name='label') #!!!
        self.queue = tf.PaddingFIFOQueue(queue_size,
                                         ['float32'],
                                         shapes=[(self.resize, self.resize, 1)])
        self.enqueue = self.queue.enqueue([self.sample_placeholder])
        self.queue_l = tf.FIFOQueue(queue_size,
                                         'int32',
                                         shapes=[])
        self.enqueue_l = self.queue_l.enqueue([self.label_placeholder])
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
            iterator = load_patient_dcm(self.data_dir, self.resize)
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
                    #TODO:  Perform quality check, make sure image is not blank
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
