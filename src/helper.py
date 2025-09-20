import numpy as np
import matplotlib.pyplot as plt

def plot_actual_vs_predicted(y_test, y_pred, label_mapping, model):
    """
    Plots side-by-side bar chart of actual vs predicted attack categories.

    Parameters:
        y_test (array-like): True labels.
        y_pred (array-like): Predicted labels.
        label_mapping (dict): Mapping of label names to indices.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(10, 6))
    categories = np.arange(len(label_mapping))
    actual_counts = np.bincount(y_test, minlength=len(label_mapping))
    pred_counts = np.bincount(y_pred, minlength=len(label_mapping))
    bar_width = 0.4

    plt.bar(categories - bar_width/2, actual_counts, width=bar_width, label='Actual', color='blue')
    plt.bar(categories + bar_width/2, pred_counts, width=bar_width, label='Predicted', color='orange')

    plt.xticks(ticks=categories, labels=[label for label in label_mapping.keys()], rotation=45)
    plt.xlabel('Attack Category')
    plt.ylabel('Frequency')
    plt.title(f'Actual vs Predicted Attack Categories: ({model})')
    plt.legend()
    plt.show()