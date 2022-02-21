import pandas as pd
from utils.EDA.data_eda import ylabeloverview
from utils.Data_Preprocess.pre_processing import get_clean_training_text


def get_data():
    training_variants = pd.read_csv(r"C:\Users\Hemang\Desktop\DataSet\Cancer_dataset\training_variants.csv")
    training_text = pd.read_csv(r"C:\Users\Hemang\Desktop\DataSet\Cancer_dataset\training_text.csv",engine="python",sep="\|\|",names=["ID","TEXT"],skiprows=1)
    return (training_variants, training_text)

def get_data_overview(training_variants, training_text):
    print("=="*30)
    print(f"Number of data points: {training_variants.shape[0]}")    
    print("=="*30)
    print(f"training_variants column names are: {training_variants.columns}")
    print("=="*30)
    print(f"training_text column names are: {training_text.columns}")
    training_variants_null = training_variants.isnull().sum()
    training_text_null = training_text.isnull().sum()
    print("=="*30)
    print(f"null data for training set")
    print(training_variants_null)
    print("=="*30)
    print(f"null data for resource set")
    print(training_text)

def get_eda_results(training_variants):
    ylabeloverview(training_variants)

def data_preprocessing(training_variants, training_text):
    training_text_clean = get_clean_training_text(training_text)
    print(training_text_clean.columns)
    result = pd.merge(training_variants, training_text,on='ID', how='left')
    result.head()
    
