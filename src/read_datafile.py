import pandas as pd

def read_data(var, text):
    '''
    Helper function for reading the training and testing data

    Returns
    =======

    data, data_text: tuple
        returns training data and data text after reading csv file

    '''
    data = pd.read_csv(var)
    data_text = pd.read_csv(text, sep="\|\|", engine="python", names=["ID", "TEXT"], skiprows=1)
    return data, data_text

