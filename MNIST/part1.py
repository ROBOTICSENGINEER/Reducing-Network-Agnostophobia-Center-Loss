class Args:
    Vanilla = False
    BG = True
    cross = True
    use_ring_loss = True
    use_lenet = False # use LeNet++
    random_model = 0 # Default
    solver = 'adam'
    lr = 0.01
    cross_entropy_loss_weight = 1.0
    ring_loss_weight = 0.0001
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

if args.Vanilla:
    model_saver = ModelCheckpoint(
                                    results_dir+'Vanilla_'+str(args.random_model)+'.h5py', monitor='val_loss', verbose=1, save_best_only=True,
                                    save_weights_only=False, mode='min', period=1)
    callbacks_list = [model_saver]
    
    if args.use_lenet:
        vanilla_lenet_pp=model_tools.LeNet()
    else:
        vanilla_lenet_pp=model_tools.LeNet_plus_plus()
    vanilla_lenet_pp.load_weights(weights_file)
    vanilla_lenet_pp.compile(optimizer=adam,loss={'softmax': 'categorical_crossentropy'},metrics=['accuracy'])
    info=vanilla_lenet_pp.fit(
                                x=[mnist.X_train],
                                y=[mnist.Y_train],
                                validation_data=[mnist.X_val,mnist.Y_val], 
                                batch_size=args.Batch_Size, epochs=args.number_of_epochs,verbose=1, callbacks=callbacks_list)
    
if args.BG:
    model_saver = ModelCheckpoint(results_dir+'BG_'+str(args.random_model)+'.h5py', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1)
    callbacks_list = [model_saver]

    X_train=np.concatenate((mnist.X_train,letters.X_train))
    tmp_mnist = np.append(mnist.Y_train,np.zeros((mnist.Y_train.shape[0],1)),1)
    tmp_neg = np.zeros((letters.X_train.shape[0],11))
    tmp_neg[:,-1]=1
    Y_train=np.concatenate((tmp_mnist,tmp_neg))
    
    class_no=np.argmax(Y_train,axis=1)
    sample_weights=np.zeros_like(class_no).astype(np.float32)
    for cls in range(11):
        sample_weights[class_no==cls]=100./len(class_no[class_no==cls])
    
    if args.use_lenet:
        raw_weights=model_tools.LeNet()
        bg_model=model_tools.LeNet(background_class=True)
    else:
        raw_weights=model_tools.LeNet_plus_plus()
        bg_model=model_tools.LeNet_plus_plus(background_class=True)
    raw_weights.load_weights(weights_file)

    for l in bg_model.layers[:-2]:
        bg_model.get_layer(l.name).set_weights(raw_weights.get_layer(l.name).get_weights())
    weights=raw_weights.get_layer('pred').get_weights()
    weights[0] = np.concatenate((weights[0],-0.0001*np.random.random_sample((weights[0].shape[0],1)) + 0.0001),axis=1)
    bg_model.get_layer('pred').set_weights(weights)
    
    
    bg_model.compile(optimizer=adam,loss={'softmax': 'categorical_crossentropy'},metrics=['categorical_accuracy'])
    info=bg_model.fit(
                        x=[X_train],
                        y=[Y_train],
                        validation_data=[mnist.X_val,np.append(mnist.Y_val,np.zeros((mnist.Y_val.shape[0],1)),1)],
                        batch_size=args.Batch_Size, epochs=args.number_of_epochs,verbose=1, callbacks=callbacks_list,sample_weight=sample_weights)

if args.cross:
    X_train,Y_train,sample_weights=model_tools.concatenate_training_data(mnist,letters.X_train,0.1)
    model_saver = ModelCheckpoint(results_dir+'Cross_'+str(args.random_model)+'.h5py', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1)
    callbacks_list = [model_saver]
    
    if args.use_lenet:
        negative_training_lenet_pp=model_tools.LeNet()
    else:
        negative_training_lenet_pp=model_tools.LeNet_plus_plus()
    negative_training_lenet_pp.load_weights(weights_file)

    negative_training_lenet_pp.compile(optimizer=adam,loss={'softmax': 'categorical_crossentropy'},metrics=['accuracy'])
    info=negative_training_lenet_pp.fit(
                                        x=[X_train],
                                        y=[Y_train],
                                        validation_data=[mnist.X_val,mnist.Y_val], 
                                        batch_size=args.Batch_Size, epochs=args.number_of_epochs,verbose=1, 
                                        callbacks=callbacks_list,sample_weight=sample_weights)

if args.use_ring_loss:
    
    X_train,Y_train,sample_weights,Y_pred_with_flags=model_tools.concatenate_training_data(mnist,letters.X_train,0.1,ring_loss=True)
    knownsMinimumMag = Input((1,), dtype='float32', name='knownsMinimumMag')
    knownsMinimumMag_ = np.ones((X_train.shape[0]))*args.Minimum_Knowns_Magnitude
    
    if args.use_lenet:
        Ring_Loss_Lenet_pp=model_tools.LeNet(ring_approach=True,knownsMinimumMag=knownsMinimumMag)
    else:
        Ring_Loss_Lenet_pp=model_tools.LeNet_plus_plus(ring_approach=True,knownsMinimumMag=knownsMinimumMag)
        
    model_saver = ModelCheckpoint(
                                    results_dir+'Ring_'+str(args.Minimum_Knowns_Magnitude)+'_'+str(args.random_model)+'.h5py',
                                    monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1
                                )
    callbacks_list = [model_saver]

    flag_placeholder=np.zeros((mnist.Y_val.shape[0],2))
    flag_placeholder[:,0]=1

    Ring_Loss_Lenet_pp.compile(
                                optimizer=adam,
                                loss={'softmax': 'categorical_crossentropy','fc':ring_loss},
                                loss_weights={'softmax': args.cross_entropy_loss_weight, 'fc': args.ring_loss_weight},
                                metrics=['accuracy'])
    info=Ring_Loss_Lenet_pp.fit(
                                    x=[X_train,knownsMinimumMag_],
                                    y=[Y_train,Y_pred_with_flags],
                                    validation_data=[
                                                        [mnist.X_val,np.ones(mnist.X_val.shape[0])*args.Minimum_Knowns_Magnitude],
                                                        [mnist.Y_val,flag_placeholder]
                                                    ],
                                    batch_size=args.Batch_Size, epochs=args.number_of_epochs,verbose=1,sample_weight=[sample_weights,sample_weights],
                                    callbacks=callbacks_list)

