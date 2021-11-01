from collections import Counter, defaultdict
import math
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def extract_dictionary_paddle(cls_text):
    dictionary = defaultdict(int)
    for index, row in cls_text.iterrows():
        for word in row['TEXT'].split():
            dictionary[word] +=1
    return dictionary

def get_text_responsecoding(df):
    text_feature_responseCoding = np.zeros((df.shape[0],9))
    for i in range(0,9):
        row_index = 0
        for index, row in df.iterrows():
            sum_prob = 0
            for word in row['TEXT'].split():
                sum_prob += math.log(((dict_list[i].get(word,0)+10 )/(total_dict.get(word,0)+90)))
            text_feature_responseCoding[row_index][i] = math.exp(sum_prob/len(row['TEXT'].split()))
            row_index += 1
    return text_feature_responseCoding

def bow_text_feature(train_df):
    text_vectorizer = CountVectorizer(min_df=3)
    train_text_feature_onehotCoding = text_vectorizer.fit_transform(train_df['TEXT'])
    # getting all the feature names (words)
    train_text_features = text_vectorizer.get_feature_names()

    # train_text_feature_onehotCoding.sum(axis=0).A1 will sum every row and returns (1*number of features) vector
    train_text_fea_counts = train_text_feature_onehotCoding.sum(axis=0).A1

    # zip(list(text_features),text_fea_counts) will zip a word with its number of times it occured
    text_fea_dict = dict(zip(list(train_text_features), train_text_fea_counts))

    print("Total number of unique words in train data :", len(train_text_features))

    return train_text_fea_counts

dict_list = []
def text_feature_implementation(train_df, train_text_features):
    # dict_list =[] contains 9 dictoinaries each corresponds to a class
    for i in range(1, 10):
        cls_text = train_df[train_df['Class'] == i]
        # build a word dict based on the words in that class
        dict_list.append(extract_dictionary_paddle(cls_text))
        # append it to dict_list

    # dict_list[i] is build on i'th  class text data
    # total_dict is buid on whole training text data
    total_dict = extract_dictionary_paddle(train_df)

    confuse_array = []
    for i in train_text_features:
        ratios = []
        max_val = -1
        for j in range(0, 9):
            ratios.append((dict_list[j][i] + 10) / (total_dict[i] + 90))
        confuse_array.append(ratios)
    confuse_array = np.array(confuse_array)
