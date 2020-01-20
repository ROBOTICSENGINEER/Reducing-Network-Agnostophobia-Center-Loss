class Args:
    ring_center = True
    use_lenet = False # use LeNet++
    random_model = 0 # Default
    solver = 'adam'
    lr = 0.01
    cross_entropy_loss_weight = 1.0
    ring_loss_weight = 0.0001
    center_loss_weight = 0.001
    Minimum_Knowns_Magnitude = 50.0
    Batch_Size  = 64 # paper = 128
    number_of_epochs  = 20 # paper = 70
    results_dir = 'LeNet++/Models/'
    
args = Args()

import sys
sys.path.insert(0, '../Tools')
import model_tools
import data_prep


"""
Setting GPU to use.
"""
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = '0'
set_session(tf.Session(config=config))


import keras
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint,LearningRateScheduler
from keras.layers import Input
from keras import backend as K
import numpy as np
import os
from keras.models import Model
from keras import losses

"""
Objectosphere loss function.
"""
def ring_loss(y_true,y_pred):
    pred=K.sqrt(K.sum(K.square(y_pred),axis=1))
    error=K.mean(K.square(
        # Loss for Knowns having magnitude greater than knownsMinimumMag
        y_true[:,0]*(K.maximum(knownsMinimumMag-pred,0.))
        # Add two losses
        +
        # Loss for unKnowns having magnitude greater than unknownsMaximumMag
        y_true[:,1]*pred
    ))
    return error

### custom loss

def zero_loss(y_true, y_pred):
    return 0.5 * K.sum(y_pred, axis=0)


mnist=data_prep.mnist_data_prep()
letters=data_prep.letters_prep()

if args.use_lenet:
    results_dir='LeNet/Models/'
    weights_file='LeNet'
else:
    results_dir='LeNet++/Models/'
    weights_file='LeNet++'

if not os.path.exists(results_dir):
    os.makedirs(results_dir)
weights_file=weights_file+'/Random_Models/model_'+str(args.random_model)+'.h5py'

if args.solver == 'adam':
    adam = Adam(lr=args.lr)
else:
    adam = SGD(lr=args.lr)

if args.ring_center:
    X_train,Y_train,sample_weights,Y_pred_with_flags=model_tools.concatenate_training_data(mnist,letters.X_train,0.1,ring_loss=True)
    knownsMinimumMag = Input((1,), dtype='float32', name='knownsMinimumMag')
    knownsMinimumMag_ = np.ones((X_train.shape[0]))*args.Minimum_Knowns_Magnitude

    if args.use_lenet:
        Ring_Loss_Lenet_pp=model_tools.LeNet(ring_approach=True,knownsMinimumMag=knownsMinimumMag)
    else:
        main_input = Input((28, 28, 1))
        aux_input = Input((10,))
        #flag_input = Input((1,))
        final_output, side_output, fc_output = model_tools.LeNet_plus_plus(main_input, aux_input,knownsMinimumMag=knownsMinimumMag)
        Ring_Loss_Lenet_pp= Model(inputs=[main_input, aux_input, knownsMinimumMag], outputs=[final_output, side_output, fc_output])
        Ring_Loss_Lenet_pp.summary()
        
    model_saver = ModelCheckpoint(
                                    results_dir+'Ring_Center_'+str(args.Minimum_Knowns_Magnitude)+'_'+str(args.random_model)+'.h5py',
                                    monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1
                                )
    callbacks_list = [model_saver]

    flag_placeholder=np.zeros((mnist.Y_val.shape[0],2))
    #flag_placeholder=np.zeros((mnist.Y_val.shape[0],1))
    flag_placeholder[:,0]=1

    dummy1 = np.zeros((X_train.shape[0], 1))
    dummy2 = np.zeros((mnist.X_val.shape[0], 1))
    
    
    Ring_Loss_Lenet_pp.compile(
                                optimizer=adam,
                                loss={'softmax':losses.categorical_crossentropy, 'centerlosslayer':zero_loss, 'fc':ring_loss},
                                loss_weights={'softmax':args.cross_entropy_loss_weight, 'centerlosslayer':args.center_loss_weight, 'fc':args.ring_loss_weight},
                                metrics=['accuracy'])
    info=Ring_Loss_Lenet_pp.fit(
                                    x=[X_train, Y_train, knownsMinimumMag_],
                                    y=[Y_train, dummy1, Y_pred_with_flags],
                                    validation_data=[
                                                        [mnist.X_val,mnist.Y_val,np.ones(mnist.X_val.shape[0])*args.Minimum_Knowns_Magnitude],
                                                        [mnist.Y_val,dummy2, flag_placeholder]
                                                    ],
                                    batch_size=args.Batch_Size, epochs=args.number_of_epochs,verbose=0,sample_weight=[sample_weights,sample_weights,sample_weights],
                                    callbacks=callbacks_list)
