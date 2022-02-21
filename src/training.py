# from utils.common import read_config
import argparse
from utils.data_mgmt import get_data
from utils.data_mgmt import get_data_overview
from utils.data_mgmt import get_eda_results
from utils.data_mgmt import data_preprocessing
from utils.data_mgmt import data_train_cv_test_split
from utils.models.random_model import random_model_result
# from models.naive_bayes import multinomial_naive_bayes
# from models.logistic_regression import logistic_regression_model
# from models.svm import svm_rbf_kernel_model
# from models.random_forest import random_forest_model


training_variants, training_text = get_data()
get_data_overview(training_variants, training_text)
get_eda_results(training_variants)
train_data_clean = data_preprocessing(training_variants, training_text)
train_df, cv_df, test_df, y_train, y_cv, y_test = data_train_cv_test_split(train_data_clean)
random_model_result(train_df, cv_df, test_df, y_train, y_cv, y_test)
# model_dict = {
#     'logistic_regression': logistic_regression_model,
#     'naive_bayes': multinomial_naive_bayes,
#     'svm': svm_rbf_kernel_model,
#     'random_forest': random_forest_model
# }

# def training(config_path, models):
#     config = read_config(config_path)
#     testing_datasize = config['params']['testing_datasize'] #TO DO from config.yaml
#     training_variants, training_text = get_data()
#     get_data_overview(training_variants, training_text)
#     get_eda_results(training_variants)
#     train_data_clean = data_preprocessing(training_variants, training_text)
#     X_tr_vec, X_test_vec, X_cv_vec, y_train, y_test, y_cv = data_vectorization_process(train_data_clean)    
    
#     model_accuracy = {}
#     for model in models:
#         model_acc = model(X_tr_vec, X_cv_vec, X_test_vec, y_train, y_cv, y_test)
#         model_accuracy[model.__name__] = model_acc
#     print(f"All the model accuracy results are {model_accuracy}")    

# if __name__ == '__main__':
#     args = argparse.ArgumentParser()
#     # args = parser.parse_args()
#     args.add_argument("--config","-c", default="config.yaml")
#     args.add_argument("--model", default="all")
#     parse_args = args.parse_args()
#     models = []
#     if parse_args.model == "all":
#         models = list(model_dict.values())
#     elif not model_dict.get(parse_args.model, False):
#         raise ValueError(f"Cannot find the model: {parse_args.model}")
#     else:
#         models = [model_dict[parse_args.model]]
#     training(parse_args.config, models)