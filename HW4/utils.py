from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from Matrix import Mat


class ConfusionMatrix(object):
    """ calculate the sensitivity and specificity to evaluate the classification model.
    """

    def __init__(self, logits, labels):
        """
        Inputs
        ------
        logits : Mat(Customized), shape=(n,1)
        labels : Mat(Customized), shape=(n,1)
        """
        assert isinstance(logits, Mat)
        assert isinstance(labels, Mat)

        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.TN = 0
        self.N = logits.shape[0]
        for i in range(logits.shape[0]):
            if int(logits[i, 0]) == 0:
                if int(labels[i, 0]) == 0:
                    self.TP += 1
                elif int(labels[i, 0]) == 1:
                    self.FP += 1
            elif int(logits[i, 0]) == 1:
                if int(labels[i, 0]) == 0:
                    self.FN += 1
                elif int(labels[i, 0]) == 1:
                    self.TN += 1

    def show_matrix(self):
        print('Confusion Matrix')
        print('N : {}'.format(self.N))
        print('True Positive : {}\tFalse Negative : {}'.format(self.TP, self.FN))
        print('False Positive : {}\tTrue Negative : {}'.format(self.FP, self.TN))

    def show_sensitivity(self):
        print('Sensitivity/Recall/HitRate')
        print(self.TP/(self.TP+self.FN))

    def show_specificity(self):
        print('Specificity')
        print(self.TN/(self.FP+self.TN))

    def show_accuracy(self):
        print('Accuracy')
        print((self.TP+self.TN)/self.N)
