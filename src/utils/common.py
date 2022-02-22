import numpy as np
import math
from collections import Counter, defaultdict

def get_gv_fea_dict(train_df, alpha, feature, df):
    value_count = train_df[feature].value_counts()
    gv_dict = dict()
        
    for i, denominator in value_count.items():
        vec = []
        for k in range(1,10):            
            cls_cnt = train_df.loc[(train_df['Class']==k) & (train_df[feature]==i)]
            vec.append((cls_cnt.shape[0] + alpha*10)/ (denominator + 90*alpha))        
        gv_dict[i]=vec
    return gv_dict

# Get Gene variation feature
def get_gv_feature(train_df, alpha, feature, df):
    
    gv_dict = get_gv_fea_dict(train_df, alpha, feature, df)
    # value_count is similar in get_gv_fea_dict
    value_count = train_df[feature].value_counts()
    
    # gv_fea: Gene_variation feature, it will contain the feature for each feature value in the data
    gv_fea = []

    for index, row in df.iterrows():
        if row[feature] in dict(value_count).keys():
            gv_fea.append(gv_dict[row[feature]])
        else:
            gv_fea.append([1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9])
#            
    return gv_fea


def extract_dictionary_paddle(cls_text):
    dictionary = defaultdict(int)
    for index, row in cls_text.iterrows():
        for word in row['TEXT'].split():
            dictionary[word] +=1
    return dictionary

def get_dict_list(train_df):
    dict_list = []
    # dict_list =[] contains 9 dictoinaries each corresponds to a class
    for i in range(1,10):
        cls_text = train_df[train_df['Class']==i]
        # build a word dict based on the words in that class
        dict_list.append(extract_dictionary_paddle(cls_text))
        # append it to dict_list

    # dict_list[i] is build on i'th  class text data
    # total_dict is buid on whole training text data
    total_dict = extract_dictionary_paddle(train_df)

    return dict_list, total_dict

#https://stackoverflow.com/a/1602964
def get_text_responsecoding(df, dict_list, total_dict):
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