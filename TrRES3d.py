import tensorflow as tf
from RESmodel import *
import numpy as np
import h5py
import math
import pydicom as dicom
import scipy.misc
import scipy.io as spio
import os
from sklearn.utils import shuffle

os.environ['CUDA_VISIBLE_DEVICES']='2'


sz=64
def extract_patches(img):
    patch_list = []
    w, h = img.shape
    for xw in np.arange(0, w-sz, 32):
        for xh in np.arange(0, h-sz, 32):
            patch_list.append(img[xw:xw+sz, xh:xh+sz])
    return patch_list

'''def prep(dataset_path):
    nums = np.random.randint(low=1, high=181, size=60)
    data = []
    label = []
    for num in nums:
        fname=dataset_path+'recon/'+'recon'+str(num)+'.h5'
        fl = h5py.File(fname, 'r')
        I = np.array(fl['recon'])
        fl.close() 
        data += extract_patches(I)

        fname=dataset_path+'Ie100/'+'mon'+str(num)+'.h5'
        ft = h5py.File(fname, 'r')
        I = np.array(ft['Ie100'])
        ft.close() 
        label += extract_patches(I)
    return data, label'''

def prep(datapath):
    patient_path=os.listdir(datapath)
    patients=np.random.choice(patient_path, 4)
    print(patients)
    data = []
    label = []
    for patient in patients:
        img_path=datapath+patient+'/2/'
        filenames= os.listdir(img_path)
        num=len(filenames)
        name_list=[]
        for filename in filenames:
            st=filename.split(".")
            dig=int(st[4])
            name_list.append(dig)
        sort_index = np.argsort(name_list)
        for ii in range(0, num):
            fn= filenames[sort_index[ii]]
            img_slice=dicom.read_file(os.path.join(img_path, fn))
            I=img_slice.pixel_array
            I=I.astype(np.float32)
            data += extract_patches(I)
        #######################
        img_path=datapath+patient+'/110/'
        filenames= os.listdir(img_path)
        name_list=[]
        for filename in filenames:
            st=filename.split(".")
            dig=int(st[4])
            name_list.append(dig)
        sort_index = np.argsort(name_list)
        for ii in range(0, num):
            fn= filenames[sort_index[ii]]
            img_slice=dicom.read_file(os.path.join(img_path, fn))
            I=img_slice.pixel_array
            I.astype(np.float32)
            label += extract_patches(I)
    print (len(data))
    print (len(label))
    return data, label


#####################
input_depth = 4
input_width = sz
input_height = sz
output_width = sz
output_height = sz
batch_size = 32
learning_rate = 1.0e-5
num_epoch = 1
###############################
X = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_depth, input_width, input_height, 1])
X0 = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_width, input_height, 1])
with tf.variable_scope('Res_Model') as scope:
    Y_ = Res_Model(X, X0)

real_data = tf.placeholder(dtype=tf.float32, shape=[batch_size, output_width, output_height, 1])
gen_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Res_Model')

#mse = tf.reduce_mean(tf.squared_difference(Y_, real_data))
mse = tf.losses.absolute_difference(Y_, real_data)
ssim = tf.reduce_mean(1.0-tf.image.ssim(Y_, real_data, 1.0))

#mse_cost = tf.sqrt(1.0 + 2 * mse) * ssim
#mse_cost=0.05*tf.sqrt(1.0+2*mse)+0.95*ssim
mse_cost=0.25*mse+0.75*ssim
#mse_cost=mse

# optimizer
lr = tf.placeholder(tf.float32, shape=[])
gen_train_op = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999).minimize(mse_cost, var_list=gen_params)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(max_to_keep = 3)
sess_path = './RESmodel3D.ckpt'
#saver.restore(sess, sess_path)

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

dataset_path = '/mnt/ext_drive1/wxcong/Yan/'
converg=np.zeros((90000,1))
iiii=-1
for iteration in range(num_epoch):
    data, label = prep(dataset_path)    
    val_lr = learning_rate #/1.0001     #/ np.sqrt(iteration + 1)
    for it in range(1):
        data, label = shuffle(data, label)
        num_batches =len(data) // batch_size       # data.shape[0] // batch_size
        for i in range(num_batches):
            iiii=iiii+1
            batch_data =  data[i*batch_size : (i+1)*batch_size]
            batch_label= label[i*batch_size : (i+1)*batch_size]
            _mse_cost, _ = sess.run([mse_cost, gen_train_op], feed_dict={real_data: normalization(batch_label),
                                                                                 X: normalization0(batch_data),
                                                                                 X0: normalization(batch_data), 
                                                                                 lr: val_lr})
            print('Epoch: %d - %d - _mse_cost: %.6f'%(iteration, it, _mse_cost))
            converg[iiii]=_mse_cost
    #saver.save(sess, './RESmodel3D.ckpt')
spio.savemat('converg.mat', {'converg': converg})
sess.close()
