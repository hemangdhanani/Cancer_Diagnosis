import pandas as pd

def read_data():
    '''
    Helper function for reading the training and testing data

    Returns
    =======

    data, data_text: tuple
        retuns training data and data text after reading csv file

    '''
    data = pd.read_csv('training_variants')
    data_text = pd.read_csv("training_text", sep="\|\|", engine="python", names=["ID", "TEXT"], skiprows=1)
    return data, data_text

