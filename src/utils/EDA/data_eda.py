import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def ylabeloverview(training_variants):
    y_label_val_count = training_variants['Class'].value_counts()
    print("=="*50)
    print(f"Class labels are : {y_label_val_count}")
    print("=="*50)    
    # # ---------------------------------------------2.2 graphical representation of approved and non-approved----------------
    # y_plot = np.array([y_approve, y_not_approve])
    # mylabels = ["Approved", "Not approved"]
    # plt.pie(y_plot, labels = mylabels)
    # plt.show()