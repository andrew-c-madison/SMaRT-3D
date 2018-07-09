'''
Train fully convolutional neural net for sematic segmentation

Instructions:
1) Set folder of train images in Train_Image_Dir
2) Set folder for ground truth labels in Train_Label_Dir. 
		Label maps should be png's with same name as corresponding image.
3) Download pretrained vgg16 model to model_path; this is done automatically from:
 		ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy or
    		https://drive.google.com/file/d/0B6njwynsu2hXZWcwX0FKTGJKRWs/view?usp=sharing
4) Set number of classes number in NUM_CLASSES
5) Set UseValidationSet=True and set validation image folders to vaildate while training
6) Run script
'''

import tensorflow as tf
import numpy as np
import Data_Reader
import BuildNetVgg16
import os
import CheckVGG16Model
import scipy.misc as misc
from sklearn.metrics import accuracy_score


# Input and output folders
Train_Image_Dir = os.path.join("Data", "Train", "images") 		# Path to Training images
Train_Label_Dir = os.path.join("Data", "Train", "labels") 		# Path to Training labels
Valid_Image_Dir = os.path.join("Data", "Validation", "images") 		# Path to Validation images 
Valid_Labels_Dir = os.path.join("Data", "Validation", "labels") 	# Path to Validation labels  
logs_dir = os.path.join("logs") 					# Path to log directory
TrainLossTxtFile = os.path.join(logs_dir, "TrainLoss.txt") 		# Path to Training loss record
ValidRecTxtFile = os.path.join(logs_dir, "ValidationRecord.txt")	# Path to Validation record


# Get pretrained model
if not os.path.exists(logs_dir): os.makedirs(logs_dir)
model_path = os.path.join("Model_Zoo", "vgg16.npy") 	# Path to pretrained vgg16 model
CheckVGG16Model.CheckVGG16(model_path) 			# Check vgg16 model, download if not present


# Define Hyperparameters 
NUM_CLASSES = 6 		# Number of classes
Batch_Size = 5			# Number of files per training iteration
Validation_Batch_Size = 5	# Number of files per validation iteration
UseValidationSet = True 	# Validation flag
UseStochasticity = True		# Apply random noise and transformations to training images
learning_rate = 1e-5		# Learning rate for Adam Optimizer
MAX_ITERATION = int(100010) 	# Max  number of training iteration

# Solver for Model
def train(loss_val, var_list):
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False)	
	grads = optimizer.compute_gradients(loss_val, var_list=var_list)
	return optimizer.apply_gradients(grads)

def main(argv=None):
	tf.reset_default_graph()
	keep_prob = tf.placeholder(tf.float32, name="keep_probabilty")	# Dropout probability

	# Placeholders for input image and labels
	image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input_image")
	GTLabel = tf.placeholder(tf.int32, shape=[None, None, None, 1], name="GTLabel")

  	# Build FCN Network
	Net =  BuildNetVgg16.BUILD_NET_VGG16(vgg16_npy_path=model_path) # Create class for the network
	Net.build(image, NUM_CLASSES, keep_prob) # Create the net and load intial weights

	# Get loss functions for neural net work one loss function for each set of labels
	Loss = -tf.reduce_sum(tf.multiply(tf.to_float(tf.one_hot(tf.squeeze(GTLabel, squeeze_dims=[3]), NUM_CLASSES)), \
				tf.log(tf.nn.softmax(Net.Prob) + 1e-12)))

	# Create solver for the net
	trainable_var = tf.trainable_variables() # Collect all trainable variables for the net
	train_op = train(Loss, trainable_var) # Create the train operation for the net

	# Create reader for training data
	TrainReader = Data_Reader.Data_Reader(Train_Image_Dir, \
				GTLabelDir=Train_Label_Dir, BatchSize=Batch_Size)
    
	# Create reader for validation data
	if UseValidationSet:
		ValidReader = Data_Reader.Data_Reader(Valid_Image_Dir, \
				GTLabelDir=Valid_Labels_Dir, BatchSize=Validation_Batch_Size) 

	# Start TensorFlow session
	sess = tf.Session() 
	print("Setting up Saver...")
	saver = tf.train.Saver()
	sess.run(tf.global_variables_initializer()) # Initialize variables
	ckpt = tf.train.get_checkpoint_state(logs_dir)
	if ckpt and ckpt.model_checkpoint_path: # Restore trained model, if it exists
		saver.restore(sess, ckpt.model_checkpoint_path)
		print("Model restored...")

	# Create files for logging progress
	f = open(TrainLossTxtFile, "w")
	f.write("Training Loss\n")
	f.write("Learning_rate\t" + str(learning_rate) + "\n")
	f.write("Batch_size\t" + str(Batch_Size) + "\n")
	f.write("Itr\tLoss")
	f.close()
	if UseValidationSet:
		f = open(ValidRecTxtFile, "w")
		f.write("Validation Record\n")
		f.write("Learning_rate\t" + str(learning_rate) + "\n")
		f.write("Batch_size\t" + str(Batch_Size) + "\n")
		f.write("Itr\tLoss\tAccuracy")
		f.close()

	# Start Training loop: Main Training
	for itr in range(MAX_ITERATION):
		if UseStochasticity:
			Images, GTLabels = TrainReader.ReadAndAugmentNextBatch() # Load images and labels
		else:
			Images, GTLabels = TrainReader.ReadNextBatchClean()
		feed_dict = {image: Images,GTLabel:GTLabels, keep_prob: 0.5}
		sess.run(train_op, feed_dict=feed_dict) # Train one cycle
		
		# Save trained model
		if itr % 500==0 and itr>0:
			print("Saving Model to file in " + logs_dir)
			saver.save(sess, os.path.join(logs_dir, "model.ckpt"), itr) # Save model
		
		# Write and display train loss
		if itr % 1==0:
        	# Calculate train loss
			feed_dict = {image: Images, GTLabel: GTLabels, keep_prob: 1}
			TLoss=sess.run(Loss, feed_dict=feed_dict)
			print("Step " + str(itr) + " Train Loss=" + str(TLoss))
        	# Write train loss to file
			with open(TrainLossTxtFile, "a") as f:
				f.write("\n"+str(itr)+"\t"+str(TLoss))
				f.close()

		# Write and display Validation Set Loss 
		if UseValidationSet and itr % 25 == 0:
			SumAcc = np.float64(0.0)
			SumLoss = np.float64(0.0)
			NBatches = np.int(np.ceil(ValidReader.NumFiles/ValidReader.BatchSize))
			print("Calculating Validation on " + str(ValidReader.NumFiles) + " Images")
			for i in range(NBatches): 
				Images, GTLabels= ValidReader.ReadNextBatchClean() # load validation data
				feed_dict = {image: Images,GTLabel: GTLabels, keep_prob: 1.0}
				
				# Calculate loss for all labels set
				TLoss = sess.run(Loss, feed_dict=feed_dict)
				SumLoss += TLoss

				# Compute validation accuracy
				pred = sess.run(Net.Pred, feed_dict={image: Images, keep_prob: 1.0})
				acc = accuracy_score(np.squeeze(GTLabels).ravel(), np.squeeze(pred).ravel())
				SumAcc += acc

			# Print validation status to console
			print("Epoch: " + str(TrainReader.Epoch))

			SumAcc/=NBatches
			SumLoss/=NBatches
			print("Validation Loss: " + str(SumLoss))
			print("Validation Accuracy: " + str(SumAcc))
			with open(ValidRecTxtFile, "a") as f:
				f.write("\n" + str(itr) + "\t" + str(SumLoss) + "\t" + str(SumAcc))
				f.close()			


if __name__ == "__main__":
	main()
	print("Finished")
