import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics._classification import accuracy_score, log_loss
import numpy as np
from scipy.sparse import hstack


def plot_confusion_matrix(test_y, predict_y):
    C = confusion_matrix(test_y, predict_y)
    A = (((C.T) / (C.sum(axis=1))).T)
    B = (C / C.sum(axis=0))

    labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # representing A in heatmap format
    print("-" * 20, "Confusion matrix", "-" * 20)
    plt.figure(figsize=(20, 7))
    sns.heatmap(C, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.show()

    print("-" * 20, "Precision matrix (Columm Sum=1)", "-" * 20)
    plt.figure(figsize=(20, 7))
    sns.heatmap(B, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.show()

    # representing B in heatmap format
    print("-" * 20, "Recall matrix (Row sum=1)", "-" * 20)
    plt.figure(figsize=(20, 7))
    sns.heatmap(A, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.show()

def predict_and_plot_confusion_matrix(train_x, train_y,test_x, test_y, clf):
    clf.fit(train_x, train_y)
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(train_x, train_y)
    pred_y = sig_clf.predict(test_x)

    # for calculating log_loss we willl provide the array of probabilities belongs to each class
    print("Log loss :",log_loss(test_y, sig_clf.predict_proba(test_x)))
    # calculating the number of data points that are misclassified
    print("Number of mis-classified points :", np.count_nonzero((pred_y- test_y))/test_y.shape[0])
    plot_confusion_matrix(test_y, pred_y)

def hstack_data(train_gene, test_gene, cv_gene, train_variation, test_variation, cv_variation, train_df, test_df, cv_df,
                train_text_oho, test_text_oho, cv_text_oho):
    train_gene_var_onehotCoding = hstack((train_gene, train_variation))
    test_gene_var_onehotCoding = hstack((test_gene, test_variation))
    cv_gene_var_onehotCoding = hstack((cv_gene, cv_variation))

    train_x_onehotCoding = hstack((train_gene_var_onehotCoding, train_text_oho)).tocsr()
    train_y = np.array(list(train_df['Class']))

    test_x_onehotCoding = hstack((test_gene_var_onehotCoding, test_text_oho)).tocsr()
    test_y = np.array(list(test_df['Class']))

    cv_x_onehotCoding = hstack((cv_gene_var_onehotCoding, cv_text_oho)).tocsr()
    cv_y = np.array(list(cv_df['Class']))
    return train_x_onehotCoding, test_x_onehotCoding, cv_x_onehotCoding
