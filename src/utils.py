import matplotlib.pyplot as plt
import numpy as np


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



