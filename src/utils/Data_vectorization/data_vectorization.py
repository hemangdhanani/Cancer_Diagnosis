import warnings
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import math
from sklearn.preprocessing import normalize
from scipy.sparse import hstack
from utils.common import get_gv_feature
from utils.common import get_text_responsecoding
from utils.common import get_dict_list


def drop_nans(data):
    data = data.dropna()
    return data

def gene_feature_oho(train_df, test_df, cv_df):
    gene_vectorizer = CountVectorizer()
    train_gene_feature_onehotCoding = gene_vectorizer.fit_transform(train_df['Gene'])
    test_gene_feature_onehotCoding = gene_vectorizer.transform(test_df['Gene'])
    cv_gene_feature_onehotCoding = gene_vectorizer.transform(cv_df['Gene'])
    return train_gene_feature_onehotCoding, test_gene_feature_onehotCoding, cv_gene_feature_onehotCoding

# Variation

def variation_feature_oho(train_df, test_df, cv_df):
    variation_vectorizer = CountVectorizer()
    train_variation_feature_onehotCoding = variation_vectorizer.fit_transform(train_df['Variation'])
    test_variation_feature_onehotCoding = variation_vectorizer.transform(test_df['Variation'])
    cv_variation_feature_onehotCoding = variation_vectorizer.transform(cv_df['Variation'])
    return train_variation_feature_onehotCoding, test_variation_feature_onehotCoding, cv_variation_feature_onehotCoding

def text_feature_oho(train_df, test_df, cv_df):
    text_vectorizer = CountVectorizer(min_df=3)
    train_text_feature_onehotCoding = text_vectorizer.fit_transform(train_df['Variation'])
    test_text_feature_onehotCoding = text_vectorizer.transform(test_df['Variation'])
    cv_text_feature_onehotCoding = text_vectorizer.transform(cv_df['Variation'])

    train_text_feature_onehotCoding = normalize(train_text_feature_onehotCoding, axis=0)
    test_text_feature_onehotCoding = normalize(test_text_feature_onehotCoding, axis=0)
    cv_text_feature_onehotCoding = normalize(cv_text_feature_onehotCoding, axis=0)

    return train_text_feature_onehotCoding, test_text_feature_onehotCoding, cv_text_feature_onehotCoding

def feature_hstack(train_gene_feature_onehotCoding, test_gene_feature_onehotCoding, cv_gene_feature_onehotCoding,train_variation_feature_onehotCoding, test_variation_feature_onehotCoding, cv_variation_feature_onehotCoding, train_text_feature_onehotCoding, test_text_feature_onehotCoding, cv_text_feature_onehotCoding,train_df, test_df, cv_df):
    train_gene_var_onehotCoding = hstack((train_gene_feature_onehotCoding,train_variation_feature_onehotCoding))
    test_gene_var_onehotCoding = hstack((test_gene_feature_onehotCoding,test_variation_feature_onehotCoding))
    cv_gene_var_onehotCoding = hstack((cv_gene_feature_onehotCoding,cv_variation_feature_onehotCoding))

    train_x_onehotCoding = hstack((train_gene_var_onehotCoding, train_text_feature_onehotCoding)).tocsr()
    train_y = np.array(list(train_df['Class']))

    test_x_onehotCoding = hstack((test_gene_var_onehotCoding, test_text_feature_onehotCoding)).tocsr()
    test_y = np.array(list(test_df['Class']))

    cv_x_onehotCoding = hstack((cv_gene_var_onehotCoding, cv_text_feature_onehotCoding)).tocsr()
    cv_y = np.array(list(cv_df['Class']))
    return train_x_onehotCoding, test_x_onehotCoding, cv_x_onehotCoding, train_y, test_y, cv_y

def gene_feature_responseCoding(train_df, test_df, cv_df):
    alpha = 1
    # train gene feature
    train_gene_feature_responseCoding = np.array(get_gv_feature(train_df, alpha, "Gene", train_df))
    # test gene feature
    test_gene_feature_responseCoding = np.array(get_gv_feature(train_df, alpha, "Gene", test_df))
    # cross validation gene feature
    cv_gene_feature_responseCoding = np.array(get_gv_feature(train_df, alpha, "Gene", cv_df))

    return train_gene_feature_responseCoding, test_gene_feature_responseCoding, cv_gene_feature_responseCoding

def variation_feature_responseCoding(train_df, test_df, cv_df):
    alpha = 1
    # train gene feature
    train_variation_feature_responseCoding = np.array(get_gv_feature(train_df, alpha, "Variation", train_df))
    # test gene feature
    test_variation_feature_responseCoding = np.array(get_gv_feature(train_df, alpha, "Variation", test_df))
    # cross validation gene feature
    cv_variation_feature_responseCoding = np.array(get_gv_feature(train_df, alpha, "Variation", cv_df))

    return train_variation_feature_responseCoding, test_variation_feature_responseCoding, cv_variation_feature_responseCoding

def text_feature_responseCoding(train_df, test_df, cv_df):
    dict_list, total_dict = get_dict_list(train_df)
    train_text_feature_responseCoding  = get_text_responsecoding(train_df, dict_list, total_dict)
    test_text_feature_responseCoding  = get_text_responsecoding(test_df, dict_list, total_dict)
    cv_text_feature_responseCoding  = get_text_responsecoding(cv_df, dict_list, total_dict)

    train_text_feature_responseCoding = (train_text_feature_responseCoding.T/train_text_feature_responseCoding.sum(axis=1)).T
    test_text_feature_responseCoding = (test_text_feature_responseCoding.T/test_text_feature_responseCoding.sum(axis=1)).T
    cv_text_feature_responseCoding = (cv_text_feature_responseCoding.T/cv_text_feature_responseCoding.sum(axis=1)).T
    
    return train_text_feature_responseCoding, cv_text_feature_responseCoding, test_text_feature_responseCoding 

def feature_hstack_responsecode(train_gene_feature_responseCoding, test_gene_feature_responseCoding, cv_gene_feature_responseCoding, train_variation_feature_responseCoding, test_variation_feature_responseCoding, cv_variation_feature_responseCoding, train_text_feature_responseCoding, cv_text_feature_responseCoding, test_text_feature_responseCoding,train_df, test_df, cv_df):
    train_gene_var_responseCoding = np.hstack((train_gene_feature_responseCoding,train_variation_feature_responseCoding))
    test_gene_var_responseCoding = np.hstack((test_gene_feature_responseCoding,test_variation_feature_responseCoding))
    cv_gene_var_responseCoding = np.hstack((cv_gene_feature_responseCoding,cv_variation_feature_responseCoding))

    train_x_responseCoding = np.hstack((train_gene_var_responseCoding, train_text_feature_responseCoding))
    train_y = np.array(list(train_df['Class']))

    test_x_responseCoding = np.hstack((test_gene_var_responseCoding, test_text_feature_responseCoding))
    test_y = np.array(list(test_df['Class']))

    cv_x_responseCoding = np.hstack((cv_gene_var_responseCoding, cv_text_feature_responseCoding))
    cv_y = np.array(list(cv_df['Class']))

    return train_x_responseCoding, cv_x_responseCoding, test_x_responseCoding, train_y, test_y, cv_y  