import numpy as np
from PIL import Image, ImageMath, ImageOps
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report
from scipy import interp
from matplotlib import cm

class ROC:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.y_img_true = np.array([])
        self.y_true = np.array([])
        self.y_score = np.array([])
        self.y_label = np.array([])
        self.report = []
        self.fpr = dict()
        self.tpr = dict()
        self.roc_auc = dict()

    def binarize_segmented_img(self, img, num_c):
        img_b = np.empty(shape=(img.shape[0], img.shape[1], num_c))
        for i in range(num_c):
            img_b[:, :, i] = (img==i+1)*1
        return img_b.ravel().reshape(num_c, img.shape[0]*img.shape[1])

    def add_data(self, img_true, img_score, img_labels):
        self.y_img_true = np.append(self.y_img_true, img_true.ravel())
        self.y_label = np.append(self.y_label, img_labels.ravel())

        y_arr_true = self.y_true.tolist()
        y_arr_score = self.y_score.tolist()
        if not y_arr_true and not y_arr_score:
            y_arr_true = [[] for i in range(self.num_classes)]
            y_arr_score = [[] for i in range(self.num_classes)]

        y_new_true = self.binarize_segmented_img(img_true, self.num_classes).tolist()
        for i,row in enumerate(y_new_true):
            y_arr_true[i].extend(row)
        self.y_true = np.asarray(y_arr_true)

        y_new_score = img_score.ravel().reshape(self.num_classes, img_score.shape[0]*img_score.shape[1])
        for i, row in enumerate(y_new_score):
            y_arr_score[i].extend(row)
        self.y_score = np.asarray(y_arr_score)

    def accuracy(self):
        if not self.y_img_true.size==0 and not self.y_label.size==0:
            self.report = classification_report(self.y_img_true, self.y_label)
            return accuracy_score(self.y_img_true, self.y_label)
        else:
            print("Data not initalized.")

    def compute_roc_curve(self):
        if not self.y_true.size==0 and not self.y_score.size==0:

            for i in range(self.num_classes):
                self.fpr[i], self.tpr[i], _ = roc_curve(self.y_true[i], self.y_score[i], pos_label=1)
                self.roc_auc[i] = auc(self.fpr[i], self.tpr[i])

            # Compute macro-average ROC curve and ROC area
            #First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([self.fpr[i] for i in range(self.num_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(self.num_classes):
                mean_tpr += interp(all_fpr, self.fpr[i], self.tpr[i])

            # Finally average it and compute AUC
            mean_tpr /= self.num_classes
            self.fpr["macro"] = all_fpr
            self.tpr["macro"] = mean_tpr
            self.roc_auc["macro"] = auc(self.fpr["macro"], self.tpr["macro"])

        else:
            print("Data not initalized.")
