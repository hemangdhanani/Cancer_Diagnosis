import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import *
from sklearn.metrics._classification import log_loss
from .model_utils import predict_and_plot_confusion_matrix


def knn_model(train_x_responseCoding, test_x_responseCoding, cv_x_responseCoding, train_y, cv_y, y_train, y_cv, y_test):
    alpha = [5, 11, 15, 21, 31, 41, 51, 99]
    cv_log_error_array = []
    for i in alpha:
        print("for alpha =", i)
        clf = KNeighborsClassifier(n_neighbors=i)
        clf.fit(train_x_responseCoding, train_y)
        sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
        sig_clf.fit(train_x_responseCoding, train_y)
        sig_clf_probs = sig_clf.predict_proba(cv_x_responseCoding)
        cv_log_error_array.append(log_loss(cv_y, sig_clf_probs, labels=clf.classes_, eps=1e-15))
        # to avoid rounding error while multiplying probabilites we use log-probability estimates
        print("Log Loss :", log_loss(cv_y, sig_clf_probs))

    fig, ax = plt.subplots()
    ax.plot(alpha, cv_log_error_array, c='g')
    for i, txt in enumerate(np.round(cv_log_error_array, 3)):
        ax.annotate((alpha[i], str(txt)), (alpha[i], cv_log_error_array[i]))
    plt.grid()
    plt.title("Cross Validation Error for each alpha")
    plt.xlabel("Alpha i's")
    plt.ylabel("Error measure")
    plt.show()

    best_alpha = np.argmin(cv_log_error_array)
    clf = KNeighborsClassifier(n_neighbors=alpha[best_alpha])
    clf.fit(train_x_responseCoding, train_y)
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(train_x_responseCoding, train_y)

    predict_y = sig_clf.predict_proba(train_x_responseCoding)
    print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",
          log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))
    predict_y = sig_clf.predict_proba(cv_x_responseCoding)
    print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",
          log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))
    predict_y = sig_clf.predict_proba(test_x_responseCoding)
    print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",
          log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))

    # Testing with best hyper parameter values

    clf = KNeighborsClassifier(n_neighbors=alpha[best_alpha])
    predict_and_plot_confusion_matrix(train_x_responseCoding, train_y, cv_x_responseCoding, cv_y, clf)
