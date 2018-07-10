# -*- coding: utf-8 -*-
"""
Created on Tue Jul 1 02:50:55 2018

@author: Gautham 
"""
#Importing all necessary files
import tensorflow as tf
import numpy as np
import Data_Reader
import TensorflowUtils as utils
import os 
import scipy.misc as misc
from PIL import Image
import matplotlib.pyplot as plt
#Specify training data path (images and annotation)
Train_Image_Dir="C:/Users/User/Mask_RCNN/samples/FCN/Data/Train/Images"
Train_Label_Dir="C:/Users/User/Mask_RCNN/samples/FCN/Data/Train/Masks_new"

#Specify Validation data path (images and annotation)
Val_Image_Dir="C:/Users/User/Mask_RCNN/samples/FCN/Data/Val/Images"
Val_Label_Dir="C:/Users/User/Mask_RCNN/samples/FCN/Data/Val/Masks_new"

#To record checkpoints and training model 
logs_dir= "C:/Users/User/Mask_RCNN/logs/"
if not os.path.exists(logs_dir): os.makedirs(logs_dir)

#Vgg model path or model URL
model_dir="C:/Users/User/Mask_RCNN/samples/FCN/Fully-convolutional-neural-network-FCN-for-semantic-segmentation-Tensorflow-implementation-master/"
MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

# Prediction & Testing 
Pred_Dir="C:/Users/User/Mask_RCNN/samples/FCN/Predictions"

# to calculate image count for training 
image_count=0
filelist=os.listdir(Train_Image_Dir)
for fichier in filelist[:]: 
    if (fichier.endswith(".png")):
        image_count=image_count+1
#Defining parameters 
learning_rate=1e-4
epochs=10
MAX_ITERATION = int(epochs*image_count)+1 #Here 419 is the number of training images
NUM_OF_CLASSESS = 2
IMAGE_SIZE = 256
Batch_Size=1

#defining the archhitecture 
def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current

    return net

def inference(image, keep_prob):
   
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(model_dir, MODEL_URL)

    mean = model_data['normalization'][0][0][0]
    
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = np.squeeze(model_data['layers'])

    processed_image = utils.process_image(image, mean_pixel)

    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, processed_image)
        conv_final_layer = image_net["conv5_3"]

        pool5 = utils.max_pool_2x2(conv_final_layer)

        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
       
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

        # now to upscale to actual image size
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")
        
        

    return tf.expand_dims(annotation_pred, dim=3), conv_t3

def train(loss_val, var_list):
    optimizer =tf.train.MomentumOptimizer(learning_rate,0.9)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)

    return optimizer.apply_gradients(grads)

def main(argv=None):
    tf.reset_default_graph()
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")
    
    pred_annotation, logits = inference(image, keep_probability)
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(annotation, squeeze_dims=[3]),
                                                                          name="entropy")))

    loss_summary = tf.summary.scalar("entropy", loss)

    trainable_var = tf.trainable_variables()
    train_op = train(loss, trainable_var)
    TrainReader = Data_Reader.Data_Reader(Train_Image_Dir,  GTLabelDir=Train_Label_Dir)
    ValReader = Data_Reader.Data_Reader(Val_Image_Dir,  GTLabelDir_Val=Val_Label_Dir)


    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    train_loss_mean=0
    epoch=0

    for itr in range(MAX_ITERATION):
        train_images, train_annotations = TrainReader.ReadNextBatchClean()
        feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85}

        sess.run(train_op, feed_dict=feed_dict)

        
        train_loss, summary_str = sess.run([loss, loss_summary], feed_dict=feed_dict)
        train_loss_mean=np.append(train_loss_mean,train_loss)
        if itr % image_count == 0 and itr>0:
            print("Saving Model to file in "+logs_dir)
            saver.save(sess, logs_dir + "model.ckpt", itr) 
        if itr%image_count==0:
            epoch=epoch+1
            print("epoch: %d, Train_loss:%g" % (epoch, np.mean(train_loss_mean)))
        
        if itr % 500 == 0:
            valid_images, valid_annotations = ValReader.ReadNextBatchClean()
            valid_loss = sess.run(loss, feed_dict={image: valid_images, annotation: valid_annotations,keep_probability: 1.0})
            print("Validation_loss: %g" % (valid_loss))
              



def test(argv=None):
    tf.reset_default_graph()
    keep_prob = tf.placeholder(tf.float32, name="keep_probabilty")  # Dropout probability
    image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input_image")
    pred_annotation, logits = inference(image, keep_prob)
    TestReader = Data_Reader.Data_Reader(Pred_Dir)
    sess1 = tf.Session() #Start Tensorflow session

    print("Setting up Saver...")
    saver = tf.train.Saver()

    sess1.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path: # if train model exist restore it
        saver.restore(sess1, ckpt.model_checkpoint_path)
        print("Model restored...")
    else:
        print("ERROR NO TRAINED MODEL IN: "+ckpt.model_checkpoint_path+" See Train.py for creating train network ")
        sys.exit()
    print("Running Predictions:")
    print("Saving output to:" + Pred_Dir)
    Images = TestReader.ReadNextBatchClean()
    LabelPred = sess1.run(pred_annotation, feed_dict={image: Images, keep_prob: 1.0})
    pred = np.squeeze(LabelPred)
    print(pred.shape)
    plt.imshow(pred)
    #misc.imsave(pred,Pred_Dir + "/Label/" + "test" + ".png")
    #pred.save(Pred_Dir + "/Label/" + "test" + ".png")

if __name__ == "__main__":
    main()
    test()
