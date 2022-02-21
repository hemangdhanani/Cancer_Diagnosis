import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer

def ylabeloverview(training_variants):
    y_label_val_count = training_variants['Class'].value_counts()
    print("=="*50)
    print(f"Class labels are : {y_label_val_count}")
    print("=="*50)

def gene_feature_eda(train_df):
    unique_genes = train_df['Gene'].value_counts()
    print('Number of Unique Genes :', unique_genes.shape[0])
    # the top 10 genes that occured most
    print(unique_genes.head(10))

    print("There are", unique_genes.shape[0] ,"different categories of genes in the train data, and they are distibuted as follows",)

    s = sum(unique_genes.values)
    h = unique_genes.values/s

    plt.plot(h, label="Histrogram of Genes")
    plt.xlabel('Index of a Gene')
    plt.ylabel('Number of Occurances')
    plt.legend()
    plt.grid()
    plt.show()

    c = np.cumsum(h)
    plt.plot(c,label='Cumulative distribution of Genes')
    plt.grid()
    plt.legend()
    plt.show()

def gene_feature_importance(train_df, cv_df, test_df, y_train, y_cv, y_test):
    gene_vectorizer = CountVectorizer()
    train_gene_feature_onehotCoding = gene_vectorizer.fit_transform(train_df['Gene'])
    test_gene_feature_onehotCoding = gene_vectorizer.transform(test_df['Gene'])
    cv_gene_feature_onehotCoding = gene_vectorizer.transform(cv_df['Gene'])

    alpha = [10 ** x for x in range(-5, 1)] # hyperparam for SGD classifier.

    cv_log_error_array=[]
    for i in alpha:
        clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)
        clf.fit(train_gene_feature_onehotCoding, y_train)
        sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
        sig_clf.fit(train_gene_feature_onehotCoding, y_train)
        predict_y = sig_clf.predict_proba(cv_gene_feature_onehotCoding)
        cv_log_error_array.append(log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))
        print('For values of alpha = ', i, "The log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

    fig, ax = plt.subplots()
    ax.plot(alpha, cv_log_error_array,c='g')
    for i, txt in enumerate(np.round(cv_log_error_array,3)):
        ax.annotate((alpha[i],np.round(txt,3)), (alpha[i],cv_log_error_array[i]))
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
    print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))
    predict_y = sig_clf.predict_proba(cv_gene_feature_onehotCoding)
    print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))
    predict_y = sig_clf.predict_proba(test_gene_feature_onehotCoding)
    print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))

def variation_feature_eda(train_df):
    unique_variation = train_df['Variation'].value_counts()
    print('Number of Unique Variation :', unique_variation.shape[0])
    # the top 10 genes that occured most
    print(unique_variation.head(10))

    print("There are", unique_variation.shape[0] ,"different categories of Variation in the train data, and they are distibuted as follows",)

    s = sum(unique_variation.values)
    h = unique_variation.values/s

    plt.plot(h, label="Histrogram of Variation")
    plt.xlabel('Index of a Variation')
    plt.ylabel('Number of Occurances')
    plt.legend()
    plt.grid()
    plt.show()

    c = np.cumsum(h)
    plt.plot(c,label='Cumulative distribution of Variation')
    plt.grid()
    plt.legend()
    plt.show()

def variation_feature_importance(train_df, cv_df, test_df, y_train, y_cv, y_test):
    variation_vectorizer = CountVectorizer()
    train_variation_feature_onehotCoding = variation_vectorizer.fit_transform(train_df['Variation'])
    test_variation_feature_onehotCoding = variation_vectorizer.transform(test_df['Variation'])
    cv_variation_feature_onehotCoding = variation_vectorizer.transform(cv_df['Variation'])

    alpha = [10 ** x for x in range(-5, 1)] # hyperparam for SGD classifier.

    cv_log_error_array=[]
    for i in alpha:
        clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)
        clf.fit(train_variation_feature_onehotCoding, y_train)
        sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
        sig_clf.fit(train_variation_feature_onehotCoding, y_train)
        predict_y = sig_clf.predict_proba(cv_variation_feature_onehotCoding)
        cv_log_error_array.append(log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))
        print('For values of alpha = ', i, "The log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

    fig, ax = plt.subplots()
    ax.plot(alpha, cv_log_error_array,c='g')
    for i, txt in enumerate(np.round(cv_log_error_array,3)):
        ax.annotate((alpha[i],np.round(txt,3)), (alpha[i],cv_log_error_array[i]))
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
    print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))
    predict_y = sig_clf.predict_proba(cv_variation_feature_onehotCoding)
    print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))
    predict_y = sig_clf.predict_proba(test_variation_feature_onehotCoding)
    print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))