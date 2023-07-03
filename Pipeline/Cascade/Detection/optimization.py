import itertools

import pandas as pd
from sklearn.metrics import f1_score, recall_score
import matplotlib.pyplot as plt

from Pipeline.evaluation import false_alarm_rate, detection_report


class GridSearch:
    """
    Class for performing grid search on the params of a detector.
    """

    def __init__(self, detector, params, train_data, train_labels):
        self.detector = detector
        self.params = params
        self.data = train_data
        self.labels = train_labels

    def optimize(self):
        f1 = 0
        best_params = None
        param_combinations = [dict(zip(self.params, v)) for v in itertools.product(*self.params.values())]
        for param in param_combinations:
            try:
                self.detector.set_params(**param)
            except ValueError:
                continue
            predictions = self.detector.detect(self.data)
            curr_f1 = f1_score(self.labels, predictions)
            if curr_f1 > f1:
                f1 = curr_f1
                best_params = param

        return best_params, f1

    def optimize_roc(self, desired_false_alarm=0.1, visual=False):
        tpr = []
        fpr = []
        param_combinations = [dict(zip(self.params, v)) for v in itertools.product(*self.params.values())]
        for param in param_combinations:
            try:
                self.detector.set_params(**param)
            except ValueError:
                continue
            predictions = self.detector.detect(self.data)
            curr_tpr = recall_score(self.labels, predictions)
            curr_fpr = false_alarm_rate(self.labels, predictions)/100
            tpr.append(curr_tpr)
            fpr.append(curr_fpr)

        if visual:
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            # add 0,0 and 1,1 points
            fpr_visual = [1] + fpr + [0]
            tpr_visual = [1] + tpr + [0]
            # add AUC
            auc = compute_auc(fpr_visual, tpr_visual)
            plt.title(f'ROC curve (AUC={auc})')
            plt.plot(fpr_visual, tpr_visual)
            plt.show()

        # Find the threshold value on the ROC curve that corresponds to the desired false alarm probability
        # by selecting the point on the curve that has an FPR closest to the desired false alarm probability
        min_index = min(range(len(fpr)), key=lambda i: abs(fpr[i] - desired_false_alarm))

        return param_combinations[min_index]


def optimize_and_evaluate(detector, params, X_train, y_train, X_test, y_test):
    optimizer = GridSearch(detector, params, X_train, y_train)

    best_params, f1 = optimizer.optimize()

    print(f'Best params: {best_params}')
    print(f'Best f1-score: {f1}')

    detector.set_params(**best_params)
    prediction = detector.detect(X_test)

    return detection_report(y_test, prediction)


def compute_auc(fpr, tpr):
    auc = pd.DataFrame({'fpr': fpr, 'tpr': tpr}).sort_values(by='fpr')
    auc = auc.sort_values(by='fpr')
    auc = auc.reset_index(drop=True)
    auc['fpr_diff'] = auc['fpr'].diff()
    auc['tpr_diff'] = auc['tpr'].diff()
    auc['auc'] = auc['fpr_diff'] * auc['tpr']
    auc = auc.dropna()
    auc = auc['auc'].sum()
    auc = round(auc, 4)

    return auc