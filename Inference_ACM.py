import tensorflow as tf
import numpy as np
import scipy.misc as misc
import sys
import BuildNetVgg16
import TensorflowUtils
import os
import Data_Reader
import OverrlayLabelOnImage as Overlay
import CheckVGG16Model
from ComputeROC import ROC
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report
from scipy import interp
from matplotlib import cm

# Directories and files
model_path = "Model_Zoo/vgg16.npy"# "Path to pretrained vgg16 model for encoder"
CheckVGG16Model.CheckVGG16(model_path)# Check if pretrained vgg16 model avialable and if not try to download it
logs_dir = "logs/"# "path to logs directory where trained model and information will be stored"
Image_Dir = "Data/Test/images/"# Test image folder
Pred_Dir = "Data/Test/predictions/" # Library where the output prediction will be written
Valid_Image_Dir = "Data/Validation/images/" # Validation images that will be used to evaluate training
Valid_Labels_Dir = "Data/Validation/labels/" #  (the  Labels are in same folder as the training set)
Valid_Pred_Dir = "Data/Validation/predictions/"

# Analysis parameters
NUM_CLASSES = 4 # Number of classes
w = 0.6# weight of overlay on image

# Main script
def main():
    # Placeholders for input image and labels
    keep_prob = tf.placeholder(tf.float32, name="keep_probabilty")  # Dropout probability
    image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input_image")  # Input image batch first dimension image number second dimension width third dimension height 4 dimension RGB

    # Build the neural network
    Net = BuildNetVgg16.BUILD_NET_VGG16(vgg16_npy_path=model_path)  # Create class instance for the net
    Net.build(image, NUM_CLASSES, keep_prob)  # Build net and load intial weights (weights before training)

    # Data reader for validation/testing images
    ValidReader = Data_Reader.Data_Reader(Image_Dir,  BatchSize=1)

    # Start Tensorflow session
    sess = tf.Session()
    print("Setting up Saver...")
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    # Load model from checkpoint
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path: # if train model exist restore it
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    else:
        print("ERROR NO TRAINED MODEL IN: "+ckpt.model_checkpoint_path+" See Train.py for creating train network ")
        sys.exit()

    # Create output directories for predicted label, one folder for each granulairy of label prediciton
    if not os.path.exists(Pred_Dir): os.makedirs(Pred_Dir)
    if not os.path.exists(Pred_Dir + "/OverLay"): os.makedirs(Pred_Dir + "/OverLay")
    if not os.path.exists(Pred_Dir + "/Label"): os.makedirs(Pred_Dir + "/Label")
    print("Running Predictions:")
    print("Saving output to:" + Pred_Dir)

    # Iterate through images and predict semantic segmentation for test set
    print("Start Predicting " + str(ValidReader.NumFiles) + " images")
    fim = 0
    while (ValidReader.itr < ValidReader.NumFiles):

        # Load image
        FileName=ValidReader.OrderedFiles[ValidReader.itr] #Get input image name
        Images = ValidReader.ReadNextBatchClean()  # load testing image

        # Predict annotation using neural net
        LabelPred = sess.run(Net.Pred, feed_dict={image: Images, keep_prob: 1.0})

        # Save predicted labels overlay on images
        misc.imsave(Pred_Dir + "/OverLay/"+ FileName, Overlay.OverLayLabelOnImage(Images[0],LabelPred[0], w)) #Overlay label on image
        misc.imsave(Pred_Dir + "/Label/" + FileName[:-4] + ".png", LabelPred[0].astype(np.uint8))
        #np.save(Pred_Dir + "/Probs/" + FileName[:-4] + ".npy", probs)

        fim += 1
        print("{:2.2f}%".format(fim * 100.0 / ValidReader.NumFiles))


    # Iterate through images and predict semantic segmentation for validation set
    if not os.path.exists(Valid_Pred_Dir + "/OverLay"): os.makedirs(Valid_Pred_Dir + "/OverLay")
    if not os.path.exists(Valid_Pred_Dir + "/Probs"): os.makedirs(Valid_Pred_Dir + "/Probs")
    if not os.path.exists(Valid_Pred_Dir + "/Label"): os.makedirs(Valid_Pred_Dir + "/Label")
    print("Validating on " + str(ValidReader.NumFiles) + " images")
    ValidReader = Data_Reader.Data_Reader(Valid_Image_Dir, GTLabelDir=Valid_Labels_Dir, BatchSize=1)
    roc = ROC(NUM_CLASSES)
    fim = 0
    while (ValidReader.itr < ValidReader.NumFiles):

        # Load image
        FileName = ValidReader.OrderedFiles[ValidReader.itr] # Get input image name
        Images, GTLabels = ValidReader.ReadNextBatchClean()  # load validation image and ground truth labels

        # Predict annotation using neural net
        LabelPred = sess.run(Net.Pred, feed_dict={image: Images, keep_prob: 1.0})

        # Get probabilities
        LabelProb = sess.run(Net.Prob, feed_dict={image: Images, keep_prob: 1.0})
        sess1 = tf.InteractiveSession()
        probs = np.squeeze(tf.nn.softmax(LabelProb).eval())


        # Import data to ROC object
        roc.add_data(np.squeeze(GTLabels), probs, np.squeeze(LabelPred))

        # Save predicted labels overlay on images
        misc.imsave(Valid_Pred_Dir + "/OverLay/"+ FileName, Overlay.OverLayLabelOnImage(Images[0],LabelPred[0], w)) #Overlay label on image
        misc.imsave(Valid_Pred_Dir + "/Label/" + FileName[:-4] + ".png", LabelPred[0].astype(np.uint8))
        np.save(Valid_Pred_Dir + "/Probs/" + FileName[:-4] + ".npy", probs)

        fim += 1
        print("{:2.2f}%".format(fim * 100.0 / ValidReader.NumFiles))

        #import pdb; pdb.set_trace()

    sess1.close()

    # Compute accuracy, precision, recall, and f-1 score
    acc = roc.accuracy()
    print(roc.report)
    print("Total Accuracy: {:3.2f}".format(acc))


main()
print("Finished")
