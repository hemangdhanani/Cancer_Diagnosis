import warnings
import pandas as pd
import numpy as np
import nltk

def drop_nans(data):
    data = data.dropna()
    return data