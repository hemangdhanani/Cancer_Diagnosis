import numpy as np
from sklearn.metrics._classification import log_loss
from .model_utils import plot_confusion_matrix

def random_model(test_df, cv_df, y_test, y_cv):
    test_data_len = test_df.shape[0]
    cv_data_len = cv_df.shape[0]

    # we create a output array that has exactly same size as the CV data
    cv_predicted_y = np.zeros((cv_data_len, 9))
    for i in range(cv_data_len):
        rand_probs = np.random.rand(1, 9)
        cv_predicted_y[i] = ((rand_probs / sum(sum(rand_probs)))[0])
    print("Log loss on Cross Validation Data using Random Model", log_loss(y_cv, cv_predicted_y, eps=1e-15))

    # Test-Set error.
    # we create a output array that has exactly same as the test data
    test_predicted_y = np.zeros((test_data_len, 9))
    op11 = len(test_predicted_y)
    op22 = len(y_test)
    for i in range(test_data_len):
        rand_probs = np.random.rand(1, 9)
        test_predicted_y[i] = ((rand_probs / sum(sum(rand_probs)))[0])
    print("==="*50)
    print("Log loss on Test Data using Random Model", log_loss(y_test, test_predicted_y, eps=1e-15))

    predicted_y = np.argmax(test_predicted_y, axis=1)
    plot_confusion_matrix(y_test, predicted_y + 1)
