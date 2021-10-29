from src.download_dataset import download_dataset
from src.read_datafile import read_data
from src.preprocess_text import nlp_runner
from src.utils import train_test_data, plot_y_distribution
from src.models.random_model import random_model
import pandas as pd
import matplotlib.pyplot as plt
import re
import time
import warnings
import numpy as np
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics._classification import accuracy_score, log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
from scipy.sparse import hstack
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from collections import Counter, defaultdict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import math
from sklearn.metrics import normalized_mutual_info_score
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")

from mlxtend.classifier import StackingClassifier

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression


var, text = download_dataset()
data, data_text = read_data(var, text)
data_text = nlp_runner(data_text)
result = pd.merge(data, data_text, on='ID', how='left')
result.loc[result['TEXT'].isnull(),'TEXT'] = result['Gene'] +' '+result['Variation']
result_dict = train_test_data(result)
train_df, cv_df, y_train, y_cv, test_df, y_test = result_dict['train_df'],result_dict['cv_df'], result_dict['y_train'],result_dict['y_cv']\
                                                  , result_dict['test_df'], result_dict['y_test']
print('Number of data points in train data:', train_df.shape[0])
print('Number of data points in test data:', test_df.shape[0])
print('Number of data points in cross validation data:', cv_df.shape[0])
print('Number of y_test ::::', y_test.shape[0])

train_class_distribution = train_df['Class'].value_counts().sort_index()
test_class_distribution = test_df['Class'].value_counts().sort_index()
cv_class_distribution = cv_df['Class'].value_counts().sort_index()

plot_y_distribution(train_class_distribution, 'train_class_distribution')
plot_y_distribution(test_class_distribution, 'test_class_distribution')
plot_y_distribution(cv_class_distribution, 'cv_class_distribution')

# y_test, y_cv
random_model(test_df, cv_df, y_test, y_cv)
print("Yeysysy bitch")
