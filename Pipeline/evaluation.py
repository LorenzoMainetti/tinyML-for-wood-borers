import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from colorama import Fore
from sklearn.metrics import RocCurveDisplay, accuracy_score, f1_score, precision_score, recall_score
from tensorflow import math


def plot_confusion_matrix(labels, y_true, y_pred):
    confusion_mtx = math.confusion_matrix(labels=y_true, predictions=y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx,
                xticklabels=labels,
                yticklabels=labels,
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()


def plot_roc_curve(y_true, y_pred):
    RocCurveDisplay.from_predictions(y_true, y_pred)
    plt.show()


def print_detection_metrics(y_true, y_pred, color=Fore.WHITE):
    test_acc = accuracy_score(y_true, y_pred) * 100
    print(color + f'Detection test set accuracy: {test_acc:.2f}%')

    false_alarm_prob = false_alarm_rate(y_true, y_pred)
    print(color + f'Detection test set false alarm prob: {false_alarm_prob}%')

    miss_detection_prob = miss_detection_rate(y_true, y_pred)
    print(color + f'Detection test set miss detection prob: {miss_detection_prob}%')


def detection_report(y_true, y_pred):
    print('\nEvaluation on the test set:')

    false_alarm_prob = false_alarm_rate(y_true, y_pred)
    print(f'False alarm prob: {false_alarm_prob}%')

    miss_detection_prob = miss_detection_rate(y_true, y_pred)
    print(f'Miss detection prob: {miss_detection_prob}%')

    d = {"label": y_true, "prediction": y_pred}
    df = pd.DataFrame(d)

    false_alarm = df[(df.prediction == True) & (df.label == False)]
    miss_detection = df[(df.prediction == False) & (df.label == True)]

    print(f'Number of False Positives: {len(false_alarm)}')
    print(f'Number of False Negatives: {len(miss_detection)}')

    print('Detector f1-score is %f%%' % (f1_score(y_true, y_pred) * 100))

    print('\nOther Metrics:')
    print('Detector accuracy is %f%%' % (accuracy_score(y_true, y_pred) * 100))
    print('Detector precision is %f%%' % (precision_score(y_true, y_pred) * 100))
    print('Detector recall is %f%%' % (recall_score(y_true, y_pred) * 100))


def false_alarm_rate(y_true, y_pred):
    # aka false positive rate = FP / FP + TN
    fp = 0
    tn = 0
    for truth, pred in zip(y_true, y_pred):
        if truth == False and pred == 1:
            fp += 1
        if truth == False and pred == 0:
            tn += 1

    fpr = (fp / (fp + tn)) * 100
    return round(fpr, 2)


def miss_detection_rate(y_true, y_pred):
    # aka false negative rate = FN / FN + TP
    fn = 0
    tp = 0
    for truth, pred in zip(y_true, y_pred):
        if truth == True and pred == 0:
            fn += 1
        if truth == True and pred == 1:
            tp += 1

    fnr = (fn / (fn + tp)) * 100
    return round(fnr, 2)
