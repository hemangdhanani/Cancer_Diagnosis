import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import seaborn as sns
from nltk.corpus import stopwords
from tqdm import tqdm
import warnings
import re

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def nlp_preprocessing(total_text, index_val, column):
    # df_res = pd.DataFrame()
    if type(total_text) is not int:        
        string_res = ""
        # replace every special char with space
        total_text = re.sub('[^a-zA-Z0-9\n]', ' ', total_text)
        # replace multiple spaces with single space
        total_text = re.sub('\s+',' ', total_text)
        # converting all the chars into lower-case.
        total_text = total_text.lower()
        
        for word in total_text.split():
        # if the word is a not a stop word then retain that word from the data
            if not word in stop_words:
                string_res += word + " "
        # print("##"*30)        
        # print(f"column is: {column}, and index is: {index_val}")
        # print(f"column type: {type(column)}, and index is: {type(index_val)}")
        # print(f"column type: {total_text}, and index is: {type(total_text)}")
        # print(f"string_res is {string_res}")
        
        
        # total_text[column][index] = string_res
        # data = pd.DataFrame({"Text":[string_res]}, index=[index_val])
        # res_df[column][index] = string_res
        # df_res = df_res.append(data)
    return string_res

def get_clean_training_text(training_text):
    for index, row in training_text.iterrows():
        if type(row['TEXT']) is str:
            total_text = nlp_preprocessing(row['TEXT'], index, 'TEXT')
            training_text['TEXT'][index] = total_text            
            # training_text['Clean Text'] = total_text
        else:
            print("there is no text description for id:",index)
    return training_text        
