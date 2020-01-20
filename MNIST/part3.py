import sys
sys.path.insert(0, '../Tools')
import data_prep
import model_tools
import visualizing_tools
import evaluation_tools
import os
import numpy as np
import pandas as pd
import keras
from keras import backend as K
from matplotlib import pyplot as plt


GPU_NO="0"
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = GPU_NO
set_session(tf.Session(config=config))

from keras.engine.topology import Layer
from keras.models import Model
from keras.utils.np_utils import to_categorical

### special layer

class CenterLossLayer(Layer):

    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(10, 2),
                                       initializer='uniform',
                                       trainable=False)
        # self.counter = self.add_weight(name='counter',
        #                                shape=(1,),
        #                                initializer='zeros',
        #                                trainable=False)  # just for debugging
        super().build(input_shape)

    def call(self, x, mask=None):

        # x[0] is Nx2, x[1] is Nx10 onehot, self.centers is 10x2
        delta_centers = K.dot(K.transpose(x[1]), (K.dot(x[1], self.centers) - x[0]))  # 10x2
        center_counts = K.sum(K.transpose(x[1]), axis=1, keepdims=True) + 1  # 10x1
        delta_centers /= center_counts
        new_centers = self.centers - self.alpha * delta_centers
        self.add_update((self.centers, new_centers), x)

        # self.add_update((self.counter, self.counter + 1), x)

        self.result = x[0] - K.dot(x[1], self.centers)
        self.result = K.sum(self.result ** 2, axis=1, keepdims=True) #/ K.dot(x[1], center_counts)
        return self.result # Nx1

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)


### custom loss

def zero_loss(y_true, y_pred):
    return 0.5 * K.sum(y_pred, axis=0)



# Loading MNIST and Letters Datasets
mnist=data_prep.mnist_data_prep()
letters=data_prep.letters_prep()

# Create Results directory if they do not exist

if not os.path.exists('LeNet++/Final_Plots'):
    os.makedirs('LeNet++/Final_Plots')
if not os.path.exists('LeNet++/DIRs'):
    os.makedirs('LeNet++/DIRs')


# The following functions performs all the plotting and storing the classification scores in a text file to be later read for DIR plotting

def analyze(model,pos_x=mnist.X_test,pos_y=mnist.labels_test,neg=letters.X_test,file_name='Vanilla_{}.{}',neg_labels='Not_MNIST', center_flag = False):
    if center_flag:        
        mnist_intermediate_output=model_tools.extract_features_center(model=model,data=pos_x,aux_lable=to_categorical(pos_y,10),layer_name=['fc','softmax','pred'])
        if neg is not None:
            neg_intermediate_output=model_tools.extract_features_center(model=model,data=neg,aux_lable=np.zeros((neg.shape[0], 10)) ,layer_name=['fc','softmax','pred'])        
        
    else:
        mnist_intermediate_output=model_tools.extract_features(model,pos_x,layer_name=['fc','softmax','pred'])
        if neg is not None:
            neg_intermediate_output=model_tools.extract_features(model,neg,layer_name=['fc','softmax','pred'])
    pred_weights=model.get_layer('pred').get_weights()[0]
    
    visualizing_tools.plotter_2D(
                                    mnist_intermediate_output[0],
                                    pos_y,
                                    neg_intermediate_output[0],
                                    final=True,
                                    file_name='LeNet++/Final_Plots/'+file_name,
                                    pos_labels='MNIST Digits',
                                    neg_labels=neg_labels,
                                    pred_weights=pred_weights
                                )
    
    visualizing_tools.plot_softmax_histogram(
                                                mnist_intermediate_output[1],
                                                neg_intermediate_output[1],
                                                file_name='LeNet++/Final_Plots/'+file_name,
                                                pos_labels='MNIST Digits',
                                                neg_labels=neg_labels
                                            )
    gt_y = np.concatenate((mnist.labels_test,np.ones(neg_intermediate_output[1].shape[0])*10),axis=0)
    pred_y = np.concatenate((mnist_intermediate_output[1],neg_intermediate_output[1]),axis=0)
    evaluation_tools.write_file_for_DIR(gt_y,
                                        pred_y,
                                        file_name=('LeNet++/DIRs/'+file_name).format(neg_labels,'txt'),
                                        num_of_known_classes=10
                                       )
    evaluation_tools.write_file_for_DIR(gt_y,
                                        pred_y,
                                        file_name=('LeNet++/DIRs/'+file_name).format(neg_labels,'txt'),
                                        feature_vector=np.concatenate((mnist_intermediate_output[0],neg_intermediate_output[0])),
                                        num_of_known_classes=10
                                       )
    
    
# Plotting DIR curves as seen in the paper
def evaluation_plotter(dataset_type,random_model_no='0'):
    evaluation_tools.process_files(DIR_filename='LeNet++/Final_Plots/'+dataset_type+'/DIR_Unknowns_'+random_model_no,
                                   files_to_process=[
                                                        'LeNet++/DIRs/'+dataset_type+'/Vanilla_'+random_model_no+'_'+dataset_type+'.txt',
                                                        'LeNet++/DIRs/'+dataset_type+'/BG_'+random_model_no+'_'+dataset_type+'.txt',
                                                        'LeNet++/DIRs/'+dataset_type+'/Cross_'+random_model_no+'_'+dataset_type+'.txt',
                                                        'LeNet++/DIRs/'+dataset_type+'/Ring_'+str(Minimum_mag_for_knowns)+'_'+random_model_no+'_'+dataset_type+'.txt',
                                                        'LeNet++/DIRs/'+dataset_type+'/Ring_Center_'+str(Minimum_mag_for_knowns)+'_'+random_model_no+'_'+dataset_type+'.txt',
                                                    ],
                                   labels=['SoftMax','Background','Entropic OpenSet Loss','ObjectSphere','Center Loss'],
                                   out_of_plot=True
                                )


# Prerequisites for Testing Objectosphere
random_model_no='0'
from keras.layers import Input
Minimum_mag_for_knowns=50.

def ring_loss(y_true,y_pred):
    pred=K.sqrt(K.sum(K.square(y_pred),axis=1))
    error=K.mean(K.square(
        # Loss for Knowns having magnitude greater than knownsMinimumMag
        y_true[:,0]*(K.maximum(knownsMinimumMag-pred,0.))
        # Add two losses
        +
        # Loss for unKnowns having magnitude greater than unknownsMaximumMag
        y_true[:,1]*(pred)
    ))
    return error
X_train,Y_train,sample_weights,Y_pred_with_flags=model_tools.concatenate_training_data(mnist,letters.X_train,0.1,ring_loss=True)
knownsMinimumMag = Input((1,), dtype='float32', name='knownsMinimumMag')
knownsMinimumMag_ = np.ones((X_train.shape[0]))*Minimum_mag_for_knowns


                           
# Comparision with CIFAR

# Loading Dataset

cifar=data_prep.cifar_prep()

dataset_type='CIFAR'
random_model_no='0'
if not os.path.exists('LeNet++/Final_Plots/'+dataset_type):
    os.makedirs('LeNet++/Final_Plots/'+dataset_type)
if not os.path.exists('LeNet++/DIRs/'+dataset_type):
    os.makedirs('LeNet++/DIRs/'+dataset_type)
 

vanilla_lenet_pp=keras.models.load_model('LeNet++/Models/Vanilla_'+random_model_no+'.h5py')
analyze(vanilla_lenet_pp,neg=cifar.images,file_name=dataset_type+'/Vanilla_'+random_model_no+'_{}.{}',neg_labels=dataset_type)

BG=keras.models.load_model('LeNet++/Models/BG_'+random_model_no+'.h5py')
analyze(BG,neg=cifar.images,file_name=dataset_type+'/BG_'+random_model_no+'_{}.{}',neg_labels=dataset_type)

negative_training_lenet_pp=keras.models.load_model('LeNet++/Models/Cross_'+random_model_no+'.h5py')
analyze(negative_training_lenet_pp,neg=cifar.images,file_name=dataset_type+'/Cross_'+random_model_no+'_{}.{}',neg_labels=dataset_type)

Ring_Loss_Lenet_pp = keras.models.load_model(('LeNet++/Models/Ring_{}_{}.h5py').format(Minimum_mag_for_knowns,random_model_no), custom_objects={'ring_loss': ring_loss})
analyze(Ring_Loss_Lenet_pp,file_name=dataset_type+'/Ring_'+str(Minimum_mag_for_knowns)+'_'+random_model_no+'_{}.{}',neg_labels=dataset_type)


Ring_Center_Loss_Lenet_pp = keras.models.load_model(('LeNet++/Models/Ring_Center_{}_{}.h5py').format(Minimum_mag_for_knowns,random_model_no), custom_objects={'ring_loss': ring_loss, 'CenterLossLayer': CenterLossLayer, 'zero_loss': zero_loss})
analyze(Ring_Center_Loss_Lenet_pp,file_name=dataset_type+'/Ring_Center_'+str(Minimum_mag_for_knowns)+'_'+random_model_no+'_{}.{}',neg_labels=dataset_type, center_flag = True)

evaluation_plotter(dataset_type,'0')
