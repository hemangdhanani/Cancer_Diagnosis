import matplotlib.pyplot as plt
import numpy as np
from ..utils import get_gv_feature
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics._classification import accuracy_score, log_loss

def gene_category_plot(train_df):
    unique_genes = train_df['Gene'].value_counts()
    print('Number of Unique Genes :', unique_genes.shape[0])
    # the top 10 genes that occured most
    print(unique_genes.head(10))
    print("Ans: There are", unique_genes.shape[0],
          "different categories of genes in the train data, and they are distributed as follows", )

    s = sum(unique_genes.values);
    h = unique_genes.values / s;
    plt.plot(h, label="Histrogram of Genes")
    plt.xlabel('Index of a Gene')
    plt.ylabel('Number of Occurances')
    plt.legend()
    plt.grid()
    plt.show()

def gene_feature_responsecode(train_df, test_df, cv_df):
    alpha = 1
    # train gene feature
    train_gene_feature_responseCoding = np.array(get_gv_feature(alpha, "Gene", train_df, train_df))
    # test gene feature
    test_gene_feature_responseCoding = np.array(get_gv_feature(alpha, "Gene", test_df, train_df))
    # cross validation gene feature
    cv_gene_feature_responseCoding = np.array(get_gv_feature(alpha, "Gene", cv_df, train_df))
    print("train_gene_feature_responseCoding is converted feature using respone coding method. The shape of gene feature:",
        train_gene_feature_responseCoding.shape)

    return train_gene_feature_responseCoding, test_gene_feature_responseCoding, cv_gene_feature_responseCoding

def gene_feature_oho(train_df, test_df, cv_df):
    gene_vectorizer = CountVectorizer()
    train_gene_feature_onehotCoding = gene_vectorizer.fit_transform(train_df['Gene'])
    test_gene_feature_onehotCoding = gene_vectorizer.transform(test_df['Gene'])
    cv_gene_feature_onehotCoding = gene_vectorizer.transform(cv_df['Gene'])
    print("train_gene_feature_onehotCoding is converted feature using one-hot encoding method. The shape of gene feature:",
        train_gene_feature_onehotCoding.shape)

    return train_gene_feature_onehotCoding, test_gene_feature_onehotCoding, cv_gene_feature_onehotCoding

def apply_model_gene_feature(train_gene_feature_onehotCoding, cv_gene_feature_onehotCoding,test_gene_feature_onehotCoding, y_train, y_cv, y_test):
    alpha = [10 ** x for x in range(-5, 1)]  # hyperparam for SGD classifier.

    cv_log_error_array = []
    for i in alpha:
        clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)
        clf.fit(train_gene_feature_onehotCoding, y_train)
        sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
        sig_clf.fit(train_gene_feature_onehotCoding, y_train)
        predict_y = sig_clf.predict_proba(cv_gene_feature_onehotCoding)
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
    clf.fit(train_gene_feature_onehotCoding, y_train)
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(train_gene_feature_onehotCoding, y_train)

    predict_y = sig_clf.predict_proba(train_gene_feature_onehotCoding)
    print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",
          log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))
    predict_y = sig_clf.predict_proba(cv_gene_feature_onehotCoding)
    print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",
          log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))
    predict_y = sig_clf.predict_proba(test_gene_feature_onehotCoding)
    print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",
          log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))


