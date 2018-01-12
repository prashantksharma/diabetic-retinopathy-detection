import itertools
import numpy as np
import matplotlib.pyplot as plt
import pickle as p

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix





def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

svm_pred=p.load(open("svm_prediction.p", "rb"))
lg_pred=p.load(open("lg_prediction.p", "rb"))
rf_pred=p.load(open("rf_prediction.p", "rb"))
actual_pred=p.load(open("actual_prediction.p", "rb"))
fig=plt.figure()
cnf_matrix_svm = confusion_matrix(actual_pred, svm_pred)
np.set_printoptions(precision=2)
fig=plt.figure()
# plot_confusion_matrix(cnf_matrix_svm, classes=['class 0','class 1','class 2'],title='Confusion matrix, without normalization')
# #plt.show()

# fig.savefig("svm_matrix_wnorm.jpg")
plot_confusion_matrix(cnf_matrix_svm, classes=['class 0','class 1','class 2'], normalize=True,title='Normalized confusion matrix')

fig.savefig("svm_matrix_norm.jpg")
#plt.show()