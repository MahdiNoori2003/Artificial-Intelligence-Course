import itertools
from python_speech_features import mfcc
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def mfcc_calculator(audio_file):
    sampling_rate, audio_data = wavfile.read(audio_file)
    mfcc_features = mfcc(audio_data, samplerate=sampling_rate, nfft=1024)
    return mfcc_features[:2100, :]


def extract_mfcc_plot_data(audio_files):
    mfccs_list = []
    sampling_rate = 0

    for audio_file in audio_files:
        mfccs = mfcc_calculator(audio_file)
        mfccs_list.append(mfccs)

    return mfccs_list


def plot_mfcc_heatmap(title, mfccs_array):
    plt.matshow(mfccs_array[0].T, cmap='viridis',
                origin='lower', aspect='auto')
    plt.title(f'MFCC Heatmap for {title}')
    plt.xlabel('Frame')
    plt.ylabel('MFCC Coefficient')
    plt.colorbar()
    plt.show()


def accuracy_score(y, y_pred):
    return np.sum(y == y_pred) / len(y)


def confusion_matrix_generator(y, y_pred, num_of_classes):
    confusion_matrix = np.zeros((num_of_classes, num_of_classes))
    for i in range(num_of_classes):
        for j in range(num_of_classes):
            confusion_matrix[i, j] = (
                np.sum((y == i) & (y_pred == j)))/(len(y)/num_of_classes)
    return confusion_matrix


def plot_confusion_matrix(cm, classes, title, cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def precision_score(y, y_pred, num_of_classes, average_method="macro"):
    if average_method == "macro":
        precisions = []
        for i in range(num_of_classes):
            tp = np.sum((y == i) & (y_pred == i))
            fp = np.sum((y != i) & (y_pred == i))
            precisions.append(tp / (tp + fp) if tp + fp != 0 else 0.0)
        return np.mean(precisions)

    elif average_method == "micro":
        tp = np.sum(y == y_pred)
        fp = np.sum(y != y_pred)
        return tp / (tp + fp) if tp + fp != 0 else 0.0

    elif average_method == "weighted":
        occurance_count = np.bincount(y)
        precisions = []
        for i in range(num_of_classes):
            tp = np.sum((y == i) & (y_pred == i))
            fp = np.sum((y != i) & (y_pred == i))
            weighted_precision = (tp / (tp + fp) if tp + fp != 0 else 0.0) * \
                (occurance_count[i] / len(y))
            precisions.append(weighted_precision)
        return np.sum(precisions)


def recall_score(y, y_pred, num_of_classes, average_method="macro"):
    if average_method == "macro":
        recalls = []
        for i in range(num_of_classes):
            tp = np.sum((y == i) & (y_pred == i))
            fn = np.sum((y == i) & (y_pred != i))
            recalls.append(tp / (tp + fn) if tp + fn != 0 else 0.0)
        return np.mean(recalls)

    elif average_method == "micro":
        tp = np.sum(y == y_pred)
        fn = np.sum(y != y_pred)
        return tp / (tp + fn) if tp + fn != 0 else 0.0

    elif average_method == "weighted":
        occurance_count = np.bincount(y)
        recalls = []
        for i in range(num_of_classes):
            tp = np.sum((y == i) & (y_pred == i))
            fn = np.sum((y == i) & (y_pred != i))
            weighted_recall = (tp / (tp + fn) if tp + fn != 0 else 0.0) * \
                (occurance_count[i] / len(y))
            recalls.append(weighted_recall)
        return np.sum(recalls)


def f1_score(y, y_pred, average_method="macro"):
    precision = precision_score(y, y_pred,
                                np.max(y)+1, average_method=average_method)
    recall = recall_score(y, y_pred,
                          np.max(y)+1, average_method=average_method)

    return 2 * (precision * recall) / (precision +
                                       recall) if precision + recall != 0 else 0.0


def accuracy_single(y_true, y_pred, label):
    y_true_copy = []
    y_pred_copy = []
    for i in range(len(y_true)):
        y_pred_copy.append(1 if y_pred[i] == label else 0)
        y_true_copy.append(1 if y_true[i] == label else 0)
    y_true_copy = np.array(y_true_copy)
    y_pred_copy = np.array(y_pred_copy)
    correct = np.sum(y_true_copy == y_pred_copy)
    total = len(y_true)
    return correct / total


def precision_single(y_true, y_pred, label):
    y_true_copy = []
    y_pred_copy = []
    for i in range(len(y_true)):
        y_pred_copy.append(1 if y_pred[i] == label else 0)
        y_true_copy.append(1 if y_true[i] == label else 0)
    y_true_copy = np.array(y_true_copy)
    y_pred_copy = np.array(y_pred_copy)
    true_positives = np.sum((y_true_copy == 1) & (y_pred_copy == 1))
    false_positives = np.sum((y_true_copy == 0) & (y_pred_copy == 1))
    return true_positives / (true_positives + false_positives)


def recall_single(y_true, y_pred, label):
    y_true_copy = []
    y_pred_copy = []
    for i in range(len(y_true)):
        y_pred_copy.append(1 if y_pred[i] == label else 0)
        y_true_copy.append(1 if y_true[i] == label else 0)
    y_true_copy = np.array(y_true_copy)
    y_pred_copy = np.array(y_pred_copy)
    true_positives = np.sum((y_true_copy == 1) & (y_pred_copy == 1))
    false_negatives = np.sum((y_true_copy == 1) & (y_pred_copy == 0))
    return true_positives / (true_positives + false_negatives)


def f1_score_single(y_true, y_pred, label):
    prec = precision_single(y_true, y_pred, label)
    rec = recall_single(y_true, y_pred, label)
    return 2 * (prec * rec) / (prec + rec)
