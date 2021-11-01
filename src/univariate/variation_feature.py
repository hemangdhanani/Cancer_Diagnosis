import matplotlib.pyplot as plt
import numpy as np
from ..utils import get_gv_feature
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics._classification import accuracy_score, log_loss

def variation_category_plot(train_df):
    unique_variations = train_df['Variation'].value_counts()
    print('Number of Unique Variation :', unique_variations.shape[0])
    # the top 10 genes that occured most
    print(unique_variations.head(10))
    print("Ans: There are", unique_variations.shape[0],
          "different categories of variations in the train data, and they are distributed as follows", )

    s = sum(unique_variations.values);
    h = unique_variations.values / s;
    plt.plot(h, label="Histogram of variations")
    plt.xlabel('Index of a variations')
    plt.ylabel('Number of Occurrences')
    plt.legend()
    plt.grid()
    plt.show()

def variation_feature_responsecode(train_df, test_df, cv_df):
    alpha = 1
    # train gene feature
    train_variation_feature_responseCoding = np.array(get_gv_feature(alpha, "Variation", train_df, train_df))
    # test gene feature
    test_variation_feature_responseCoding = np.array(get_gv_feature(alpha, "Variation", test_df, train_df))
    # cross validation gene feature
    cv_variation_feature_responseCoding = np.array(get_gv_feature(alpha, "Variation", cv_df, train_df))
    print("train_gene_feature_responseCoding is converted feature using respone coding method. The shape of gene feature:",
        train_variation_feature_responseCoding.shape)

    return train_variation_feature_responseCoding, test_variation_feature_responseCoding, cv_variation_feature_responseCoding

def variation_feature_oho(train_df, test_df, cv_df):
    variation_vectorizer = CountVectorizer()
    train_variation_feature_onehotCoding = variation_vectorizer.fit_transform(train_df['Gene'])
    test_variation_feature_onehotCoding = variation_vectorizer.transform(test_df['Gene'])
    cv_variation_feature_onehotCoding = variation_vectorizer.transform(cv_df['Gene'])
    print("train_gene_feature_onehotCoding is converted feature using one-hot encoding method. The shape of gene feature:",
        train_variation_feature_onehotCoding.shape)

    return train_variation_feature_onehotCoding, test_variation_feature_onehotCoding, cv_variation_feature_onehotCoding

def apply_model_variation_feature(train_variation_feature_onehotCoding, test_variation_feature_onehotCoding, cv_variation_feature_onehotCoding, y_train, y_cv, y_test):
    alpha = [10 ** x for x in range(-5, 1)]

    cv_log_error_array = []
    for i in alpha:
        clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)
        clf.fit(train_variation_feature_onehotCoding, y_train)

        sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
        sig_clf.fit(train_variation_feature_onehotCoding, y_train)
        predict_y = sig_clf.predict_proba(cv_variation_feature_onehotCoding)

        cv_log_error_array.append(log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))
        print('For values of alpha = ', i, "The log loss is:",
              log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

    fig, ax = plt.subplots()
    ax.plot(alpha, cv_log_error_array, c='g')
    for i, txt in enumerate(np.round(cv_log_error_array, 3)):
        ax.annotate((alpha[i], np.round(txt, 3)), (alpha[i], cv_log_error_array[i]))
    plt.grid()
    plt.title("Cross Validation Error for each alpha")
    plt.xlabel("Alpha i's")
    plt.ylabel("Error measure")
    plt.show()

    best_alpha = np.argmin(cv_log_error_array)
    clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)
    clf.fit(train_variation_feature_onehotCoding, y_train)
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(train_variation_feature_onehotCoding, y_train)

    predict_y = sig_clf.predict_proba(train_variation_feature_onehotCoding)
    print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",
          log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))
    predict_y = sig_clf.predict_proba(cv_variation_feature_onehotCoding)
    print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",
          log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))
    predict_y = sig_clf.predict_proba(test_variation_feature_onehotCoding)
    print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",
          log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))

    print("..............Current Done............")
