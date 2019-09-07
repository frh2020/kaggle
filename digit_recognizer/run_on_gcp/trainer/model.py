
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import shutil
import os


from sklearn.model_selection import train_test_split

import tensorflow as tf


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import TensorBoard

tf.logging.set_verbosity(tf.logging.INFO)

HEIGHT = 28
WIDTH = 28
NCLASSES = 10

def cnn_model(features, labels, mode, params):

    X = features['image']

    c1 = tf.layers.conv2d(inputs = X, filters = 16, kernel_size = 5, strides = 1, padding = 'same', activation = tf.nn.relu)  # shape = (batch_size, HEIGHT, WIDTH, nfil1)

    p1 = tf.layers.max_pooling2d(inputs=c1, pool_size = 2, strides = 2)                                              # shape = (batch_size, HEIGHT // 2, WIDTH // 2, nfil1)
    
    c2 = tf.layers.conv2d(inputs = p1, filters = 32, kernel_size = 5, strides = 1, padding = 'same', activation = tf.nn.relu)    # shape = (batch_size, HEIGHT // 2, WIDTH // 2, nfil2)
    
    p2 = tf.layers.max_pooling2d(inputs=c2, pool_size = 2, strides = 2)                                                       # shape = (batch_size, HEIGHT // 4, WIDTH // 4, nfil2)

    c3 = tf.layers.conv2d(inputs = p2, filters = 64, kernel_size = 3, strides = 1, padding = 'same', activation = tf.nn.relu)    # shape = (batch_size, HEIGHT // 2, WIDTH // 2, nfil2)

    p3 = tf.layers.max_pooling2d(inputs=c3, pool_size = 2, strides = 2)     

    p3flat =  tf.reshape(tensor = p3, shape= [-1, p3.shape[1] * p3.shape[2] * p3.shape[3]])                                                                  # shape = (batch_size, HEIGHT // 4 * WIDTH // 4 * nfil2)

    h3 = tf.layers.dense(inputs = p3flat, units = 64, activation = tf.nn.relu)
    
    h3d = tf.layers.dropout(inputs = h3, rate = 0.25, training = (mode == tf.estimator.ModeKeys.TRAIN))    
    
    ylogits = tf.layers.dense(inputs = h3d, units = NCLASSES, activation = None)
  

    probabilities = tf.nn.softmax(logits = ylogits)
    
    class_ids = tf.cast(x = tf.argmax(input = probabilities, axis = 1), dtype = tf.uint8)

    global_step = tf.train.get_global_step()

    learning_rate=tf.train.exponential_decay(
                  params["learning_rate"],
                  global_step,
                  decay_steps=params["decay_steps"],
                  decay_rate=0.95,
                  staircase=True)
    
    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
        
        global_step = tf.train.get_global_step()

        learning_rate=tf.train.exponential_decay(
                  params["learning_rate"],
                  global_step,
                  decay_steps=params["decay_steps"],
                  decay_rate=0.95,
                  staircase=True)
        
        loss = tf.reduce_mean(input_tensor = tf.nn.softmax_cross_entropy_with_logits(logits = ylogits, labels = labels))
        
        accuracy=tf.metrics.accuracy(labels = tf.argmax(input = labels, axis = 1), predictions = class_ids)
        
        if mode == tf.estimator.ModeKeys.TRAIN :
        
            optimizer=tf.train.AdamOptimizer(learning_rate)
            
            gradients = optimizer.compute_gradients(loss=loss)

           # last_grad_norm=0;
           # for g,v in gradients:
           #     if 'last' in v.name :
           #         last_grad_norm = tf.sqrt(tf.reduce_mean(g**2))
           #     break
            
            train_op = optimizer.apply_gradients(gradients, global_step=global_step)
               
            eval_metric_ops = None
    
            with tf.name_scope('performance'):
                tf.summary.scalar('loss', loss)
                tf.summary.scalar('learning_rate', learning_rate)
                tf.summary.scalar('accuracy', accuracy[1])

                   
        else:
            
            train_op = None
            eval_metric_ops =  {"accuracy": accuracy}
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        loss = None
        train_op = None
        eval_metric_ops = None

    predictions = {"key": tf.identity(features['key']) , "probabilities": probabilities, "class_ids": class_ids}
    
    return tf.estimator.EstimatorSpec(
        mode = mode,
        predictions =predictions,
        loss = loss,
        train_op = train_op,
        eval_metric_ops = eval_metric_ops,
        export_outputs = {"predictions": tf.estimator.export.PredictOutput(predictions)}
   )

def serving_input_fn():

    feature_placeholders = {"image": tf.placeholder(dtype = tf.float32, shape = [None, HEIGHT, WIDTH]),"key": tf.placeholder(dtype = tf.int32, shape=[None,])}

    features = {"image": tf.expand_dims(input = feature_placeholders["image"], axis = -1),"key": feature_placeholders["key"]} 
    return tf.estimator.export.ServingInputReceiver(features =  features, receiver_tensors =  feature_placeholders)
  
def Input_data(input_dir):
	train = pd.read_csv(input_dir+'/train.csv')
    
	train_images = train.drop(labels = ["label"],axis = 1)
	train_labels = train["label"].astype('int32')

	train_images = train_images.to_numpy().astype('float32')/255.0
	train_images = train_images.reshape(-1,HEIGHT,WIDTH,1)

	train_labels = tf.keras.utils.to_categorical(train_labels, num_classes = NCLASSES)
    
	return train_images, train_labels
  
def train_and_evaluate(output_dir, input_dir, hparams):
	tf.summary.FileWriterCache.clear() # ensure filewriter cache is clear for TensorBoard events file
    
	train_images, train_labels = Input_data(input_dir)
    
	X_train, X_val, Y_train, Y_val = train_test_split(train_images, train_labels, test_size = 0.1)
    
	key_val=np.arange(len(Y_val)).reshape([-1,])

	key_train=np.arange(len(Y_train)).reshape([-1,])
    
	datagen = ImageDataGenerator(
        rotation_range=10,  
        zoom_range = 0.10,  
        width_shift_range=0.1, 
        height_shift_range=0.1)

	datagen.fit(X_train)
    
	hparams['decay_steps'] = len(Y_train)// hparams['batch_size']

	estimator = tf.estimator.Estimator(
        model_fn = cnn_model,
        model_dir = output_dir,
        params = hparams)

	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
	    x = {'image': X_val, 'key': key_val},
	    y = Y_val,
	    batch_size = hparams['batch_size'],
	    num_epochs = 1,
	    shuffle = False,
	    queue_capacity = 5000 )
    

	for ep in range(hparams['epochs']):
        
		(X_train_da, Y_train_da)  = datagen.flow(X_train, Y_train, batch_size = len(Y_train)).next()
        
              
		train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x = {'image':X_train_da, 'key': key_train},
            y = Y_train_da,
            batch_size = hparams['batch_size'],
            num_epochs = 1,
            shuffle = True,
            queue_capacity = 5000
            )
        
		estimator.train(input_fn=train_input_fn)
        
		eval_results = estimator.evaluate(input_fn=eval_input_fn)
        
		print("epoch {0:02d}: {1} \n".format(ep, eval_results)) 
    
	estimator.export_saved_model(output_dir+"export",  serving_input_fn) 