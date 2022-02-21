import pandas as pd
# import sklearn
from sklearn.model_selection import train_test_split
from utils.EDA.data_eda import ylabeloverview
from utils.EDA.data_eda import gene_feature_eda
from utils.EDA.data_eda import variation_feature_eda
from utils.EDA.data_eda import gene_feature_importance
from utils.EDA.data_eda import variation_feature_importance
from utils.Data_Preprocess.pre_processing import get_clean_training_text
from utils.Data_vectorization.data_vectorization import drop_nans


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

def data_preprocessing(training_variants, training_text):
    training_text_clean = get_clean_training_text(training_text)
    print(training_text_clean.columns)
    df_result = pd.merge(training_variants, training_text,on='ID', how='left')
    print(df_result.head())
    df_result.loc[df_result['TEXT'].isnull(),'TEXT'] = df_result['Gene'] +' '+df_result['Variation']
    return df_result
    
def data_train_cv_test_split(train_data_clean):
    train_data_without_nan = drop_nans(train_data_clean)
    y_true = train_data_clean['Class'].values
    train_data_clean.Gene = train_data_clean.Gene.str.replace('\s+', '_')
    train_data_clean.Variation = train_data_clean.Variation.str.replace('\s+', '_')
    X_train, test_df, y_train, y_test = train_test_split(train_data_clean, y_true, stratify=y_true, test_size=0.2)
    train_df, cv_df, y_train, y_cv = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2)
    print('Number of data points in train data:', train_df.shape[0])
    print('Number of data points in test data:', test_df.shape[0])
    print('Number of data points in cross validation data:', cv_df.shape[0])
    return train_df, cv_df, test_df, y_train, y_cv, y_test

def get_eda_results(training_variants, train_df, cv_df, test_df, y_train, y_cv, y_test):
    ylabeloverview(training_variants)
    gene_feature_eda(train_df)
    gene_feature_importance(train_df, cv_df, test_df, y_train, y_cv, y_test)
    variation_feature_eda(train_df)
    variation_feature_importance(train_df, cv_df, test_df, y_train, y_cv, y_test)    