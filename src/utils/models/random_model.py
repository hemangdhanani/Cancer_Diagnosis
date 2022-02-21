import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from utils.models.model_utils import plot_confusion_matrix

def random_model_result(train_df, cv_df, test_df, y_train, y_cv, y_test):
    print("random model starts....")
    test_data_len = test_df.shape[0]
    cv_data_len = cv_df.shape[0]

    # we create a output array that has exactly same size as the CV data
    cv_predicted_y = np.zeros((cv_data_len,9))
    for i in range(cv_data_len):
        rand_probs = np.random.rand(1,9)
        cv_predicted_y[i] = ((rand_probs/sum(sum(rand_probs)))[0])
    print("Log loss on Cross Validation Data using Random Model",log_loss(y_cv,cv_predicted_y, eps=1e-15))


    # Test-Set error.
    #we create a output array that has exactly same as the test data
    test_predicted_y = np.zeros((test_data_len,9))
    for i in range(test_data_len):
        rand_probs = np.random.rand(1,9)
        test_predicted_y[i] = ((rand_probs/sum(sum(rand_probs)))[0])
    print("Log loss on Test Data using Random Model",log_loss(y_test,test_predicted_y, eps=1e-15))

    predicted_y =np.argmax(test_predicted_y, axis=1)
    plot_confusion_matrix(y_test, predicted_y+1)
    print("...random model end")