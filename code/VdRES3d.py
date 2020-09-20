import tensorflow as tf
from RESmodel import *
import numpy as np
import h5py
import pydicom as dicom
import time
import os
import scipy.misc
import scipy.io as spio
import pickle as pkl
from sklearn.utils import shuffle

#os.environ['CUDA_VISIBLE_DEVICES']='1'
input_depth = 4
input_width = 512
input_height = 512

# generator networks
X = tf.placeholder(dtype=tf.float32, shape=[1, input_depth, input_width, input_height, 1])
X0 = tf.placeholder(dtype=tf.float32, shape=[1, input_width, input_height, 1])
with tf.variable_scope('Res_Model') as scope:
    Y = Res_Model(X, X0)

def normalization0(x):
    water=0.2
    air=0.0002
    x= np.subtract(x,1024.0)
    x=np.multiply((water-air), x) 
    x=np.divide(x,1000.0)+water
    x[x<0.0] = 0.0
    tmp=np.ones(np.shape(x)) 
    Y = np.expand_dims(tmp, axis=1)
    for k in range(3):
        tmp=np.multiply(tmp, x)       
        Y=np.concatenate((Y, np.expand_dims(tmp, axis=1)), axis=1)
    Y = np.expand_dims(Y, axis=4)
    return Y.astype(np.float32)

def normalization(x):
    water=0.2
    air=0.0002
    x= np.subtract(x,1024.0)
    x=np.multiply((water-air), x) 
    x=np.divide(x,1000.0)+water
    x[x<0.0] = 0.0
    x = np.expand_dims(x, axis=3)
    return x.astype(np.float32)

# restore trained model
sess = tf.Session()
saver = tf.train.Saver()
sess_path = './RESmodel3D.ckpt'
saver.restore(sess, sess_path)


img_path = '/mnt/ext_drive1/wxcong/Yantest/001/2/'
#slices= os.listdir(img_path)
#slice = np.random.choice(slices, 1)[0]

slice='YangZhanKui.CT.Abdomen_15DES.2.165.2020.06.20.12.33.53.21.77464910.dcm'
print(slice)
img_slice=dicom.read_file(os.path.join(img_path, slice))
I=img_slice.pixel_array
I=I.astype(np.float32)

img_pixel = np.expand_dims(I, axis=0)
output_img = sess.run(Y, feed_dict = {X : normalization0(img_pixel), X0: normalization(img_pixel)})
output_img = np.squeeze(output_img)
spio.savemat('/home/wxcong/XEnergy/E100/Yandata/slice110md.mat', {'img': output_img})



'''
fname = '/home/wxcong/XEnergy/Ie100/recon_GE.h5'
#slices= os.listdir(img_path)
#for slice in slices:
#    print(slice)
#    fname=dataset_path+'Iatt/'+'att'+str(num)+'.h5'
fp = h5py.File(fname, 'r')
img_slice = np.array(fp['recon'])
fp.close() 
img_pixel = np.expand_dims(img_slice, axis=0)
output_img = sess.run(Y, feed_dict = {X : normalization0(img_pixel), X0: normalization(img_pixel)})
output_img = np.squeeze(output_img)
spio.savemat('/home/wxcong/XEnergy/Ie100/recon100_GE'+'.mat', {'img': output_img})
'''
#slice+'.mat', {'img': output_img})
 
'''
for slice in slices:
    print(slice)
    ds = dicom.dcmread(os.path.join(img_path, slice))
    img_slice=dicom.read_file(os.path.join(img_path, slice))
    img_pixel=img_slice.pixel_array
    img_pixel = np.expand_dims(img_pixel, axis=0)
    #img_pixel = np.expand_dims(img_pixel, axis=3) # 1 x 512 x 512 x 1
    output_img = sess.run(Y, feed_dict = {X : normalization(img_pixel)})
    output_img = np.squeeze(output_img)
    ds.PixelData = np.int16(output_img)
    (ds.Rows, ds.Columns) = np.shape(output_img)
    #dicom.write_file('/home/congw/OpeCom/results/AIrecon%s.dcm'%str(s).zfill(2), ds)
    dicom.write_file('/home/congw/OpeCom/results/'+slice+'.dcm', ds)'''
sess.close()