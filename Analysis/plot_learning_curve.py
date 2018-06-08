# Note: Monitor GPU activity at 0.5 s intervals: watch -n 0.5 nvidia-smi

import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

def itr2epoch(i,b,n):
	return i*b/n

def tick_function(i,b,n):
    return ["%d" % z for z in itr2epoch(i,b,n)]


# Input data files
Train_Loss_txt = os.path.join("..", "logs", "TrainLoss.txt") 			# Path to Training Loss
Train_Image_Dir = os.path.join("..", "Data", "Train", "images") 		# Path to Training images
Valid_Rec_txt = os.path.join("..", "logs", "ValidationRecord.txt")		# Path to Validation Loss and Accuracy

# Set parameters
num_examples = len([f for f in os.listdir(Train_Image_Dir)])

# Read header data to get batch size
with open(Train_Loss_txt) as f:
	for i, line in enumerate(f):
		if i==2:
			s = line.replace("\t", " ").replace("\n", "").split()
			batch_size = int(s[1])

# Read training and validation data
Training_loss = np.loadtxt(Train_Loss_txt, skiprows=4)
Validation_data = np.loadtxt(Valid_Rec_txt, skiprows=4)

# Load Data into arrays
xT = Training_loss[:,0]
yT = Training_loss[:,1]

xV = Validation_data[:,0]
yV = Validation_data[:,1]

xA = Validation_data[:,0]
yA = Validation_data[:,2]

# Set up axes
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax3 = ax1.twiny()

# Plot data
lns1 = ax1.semilogy(xT, yT, 'r-', alpha=0.3, label='Training Loss')
lns2 = ax1.semilogy(xV, yV, 'r-', label='Validation Loss')
lns3 = ax2.plot(xA, yA, 'b--', label='Validation Accuracy')

# Setup 2nd x-axis
ax3.spines["bottom"].set_position(("axes", -0.15))
make_patch_spines_invisible(ax3)
ax3.spines["bottom"].set_visible(True)
ax3.xaxis.set_ticks_position("bottom")
ax3.xaxis.set_label_position("bottom")
ax3.spines["bottom"].set_position(("axes", -0.15))
ax3.set_frame_on(True)
ax3.patch.set_visible(False)
for sp in ax3.spines.items():
	sp[1].set_visible(False)
ax3.spines["bottom"].set_visible(True)

# Label axes
ax1.set_xlabel('Iteration')
ax3.set_xlabel("Epoch")
ax1.set_ylabel('Loss', color='r')
ax2.set_ylabel('Accuracy', color='b')
ax2.set_ylim(0,1)
xlims = ax1.get_xlim() 
ax3.set_xlim(itr2epoch(xlims[0],batch_size, num_examples), itr2epoch(xlims[1],batch_size, num_examples))

# Add legend
lns = lns1 + lns2 + lns3
labs = [l.get_label() for l in lns]
plt.legend(lns, labs, loc=3)

# Show plot
fig.tight_layout()
plt.show()
