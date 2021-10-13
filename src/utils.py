import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def plot_y_distribution(df):
    class_distribution = df['Class'].value_counts().sort_index()
    my_colors = 'rgbkymc'
    class_distribution.plot(kind='bar')
    plt.xlabel('Class')
    plt.ylabel('Data points per Class')
    plt.title('Distribution of yi in train data')
    plt.grid()
    plt.show()

    # -(train_class_distribution.values): the minus sign will give us in decreasing order
    sorted_yi = np.argsort(-class_distribution.values)
    for i in sorted_yi:
        print('Number of data points in class', i + 1, ':', class_distribution.values[i], '(',
              np.round((class_distribution.values[i] / df.shape[0] * 100), 3), '%)')

def onehot_encode(train_df, test_df, cv_df):
    # one-hot encoding of variation feature.
    variation_vectorizer = CountVectorizer()
    train_variation_feature_onehotCoding = variation_vectorizer.fit_transform(train_df['Variation'])
    test_variation_feature_onehotCoding = variation_vectorizer.transform(test_df['Variation'])
    cv_variation_feature_onehotCoding = variation_vectorizer.transform(cv_df['Variation'])
    return (train_variation_feature_onehotCoding, test_variation_feature_onehotCoding, cv_variation_feature_onehotCoding)


def get_gv_fea_dict(alpha, feature, df, train_df):
    value_count = train_df[feature].value_counts()
    gv_dict = dict()
    for i, denominator in value_count.items():
        vec = []
        for k in range(1, 10):
            cls_cnt = train_df.loc[(train_df['Class'] == k) & (train_df[feature] == i)]
            vec.append((cls_cnt.shape[0] + alpha * 10) / (denominator + 90 * alpha))
        gv_dict[i] = vec
    return gv_dict

def get_gv_feature(alpha, feature, df, train_df):
    gv_dict = get_gv_fea_dict(alpha, feature, df)
    value_count = train_df[feature].value_counts()
    gv_fea = []

    for index, row in df.iterrows():
        if row[feature] in dict(value_count).keys():
            gv_fea.append(gv_dict[row[feature]])
        else:
            gv_fea.append([1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9])

    return gv_fea

def response_code(train_df, test_df, cv_df):
    # alpha is used for laplace smoothing
    alpha = 1
    # train gene feature
    train_variation_feature_responseCoding = np.array(get_gv_feature(alpha, "Variation", train_df))
    # test gene feature
    test_variation_feature_responseCoding = np.array(get_gv_feature(alpha, "Variation", test_df))
    # cross validation gene feature
    cv_variation_feature_responseCoding = np.array(get_gv_feature(alpha, "Variation", cv_df))
    return (train_variation_feature_responseCoding, test_variation_feature_responseCoding, cv_variation_feature_responseCoding)
