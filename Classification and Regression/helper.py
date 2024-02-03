import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_corr_scatter_hexbin(col, df, target):
    fig, axs = plt.subplots(ncols=2, figsize=(12, 4))
    plt.suptitle(col)
    axs[0].scatter(df[col], df[target])
    axs[1].hexbin(df[col], df[target], gridsize=20, cmap="Blues")
    plt.subplots_adjust(wspace=0.3)
    plt.show()


def missing_values_count_and_percentage(df):
    nan_values_count = df.isna().sum()
    nan_values_percent = nan_values_count / len(df)
    nan_values = pd.concat([nan_values_count, nan_values_percent], axis=1, keys=[
                           "Missing Count", "Ratio"])
    return nan_values


def plot_simple_reg_line(x, y, slope, intercept, feature_name):
    plt.scatter(x, y)
    regression_x = np.linspace(x.min(), x.max(), 100)
    regression_y = slope * regression_x + intercept

    plt.plot(regression_x, regression_y, color='red',
             label=f"y = {slope:0.2f} * x + {intercept:0.2f}")

    plt.xlabel(feature_name)
    plt.ylabel('NumPurchases')

    plt.legend()
    plt.show()


def plot_confusion_matrix(preds, y_test, title):
    cm = confusion_matrix(y_test, preds)
    cm_display = ConfusionMatrixDisplay(cm)
    cm_display.plot(cmap='Blues')
    plt.title(f'{title} Confusion Matrix')
    plt.show()
