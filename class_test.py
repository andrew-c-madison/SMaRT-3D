from ComputeROC import ROC
import numpy as np
from PIL import Image, ImageMath, ImageOps
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report
from scipy import interp
from matplotlib import cm

def main():
	NUM_CLASSES = 4
	roc = ROC(NUM_CLASSES)
	name_true = "/media/andy/Elements/Camp_Cook/Classified_Imagery/IMG_9672_truth.png"
	name_labels = "/media/andy/Elements/Camp_Cook/Classified_Imagery/Labels/IMG_9672_labels.npy"
	name_score = "/media/andy/Elements/Camp_Cook/Classified_Imagery/Probabilities/IMG_9672_probabilities.npy"

	# Import ground truth images
	img_true = Image.open(name_true)
	img_true = np.array(img_true.convert('L'))

	# Import probabilities from Classification
	img_score = np.load(name_score)
	img_label = np.load(name_labels)

	# Import data
	roc.add_data(img_true, img_score, img_label)

	# Compute accuracy, precision, recall, and f-1 score
	acc = roc.accuracy()
	print(roc.report)
	print("Total Accuracy: {:3.2f}".format(acc))

	# Compute ROC curve and plot
	roc.compute_roc_curve()
	for i in range(roc.num_classes):
	    plt.plot(roc.fpr[i], roc.tpr[i], c=cm.bone_r((i+0.2)/3.0),label=r'ROC$_{0}$, AUC={1:0.2f}'''.format(i, roc.roc_auc[i]))

	# Plot macro-average
	plt.plot(roc.fpr["macro"], roc.tpr["macro"],label=r'$\langle$ROC$\rangle$, AUC={0:0.2f}'''.format(roc.roc_auc["macro"]), color='red', linestyle='--')

	# Plot random guess line
	x = np.linspace(0,1,100)
	y = x
	plt.plot(x,y,'k:')
	text = plt.gca().annotate("Random", xy=(0.5, 0.45), xytext=(0, 0),
	                       textcoords='offset points',
	                       size=10, color='black',
	                       horizontalalignment='center',
	                       verticalalignment='center', rotation=45)

	# Labels
	plt.gca().set_aspect('equal', 'box')
	plt.xlim(-0.1, 1.1)
	plt.ylim(-0.1, 1.1)
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.legend(loc="lower right", fontsize=10)
	plt.show()


main()
