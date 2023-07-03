import numpy as np
import time


class SEFR:
    """
    SEFR is a fast, linear-time classifier to be used mainly on, but not limited to, ultra-low power devices.
    The data should be normalized in the range [0, n], and its labels should be 0 and 1.
    For training and prediction, fit() and predict() methods are used, the same way as in other classifiers.
    """
    def fit(self, train_data, train_target):
        pass

    def predict(self, test_data):
        pass


class SEFRBinaryClass(SEFR):
    def __init__(self):
        self.weights = []
        self.bias = 0

    def fit(self, train_data, train_target):
        """
        This is used for training the classifier on data.
        Parameters
        ----------
        train_data : float, either list or numpy array
            are the main data in DataFrame
        train_target : integer, numpy array
            labels, should consist of 0s and 1s
        """
        X = np.array(train_data, dtype="float32")
        y = np.array(train_target, dtype="int32")

        # pos_labels are those records where the label is positive
        # neg_labels are those records where the label is negative
        pos_labels = np.sign(y) == 1
        neg_labels = np.invert(pos_labels)

        # pos_indices are the data where the labels are positive
        # neg_indices are the data where the labels are negative
        pos_indices = X[pos_labels, :]
        neg_indices = X[neg_labels, :]

        # avg_pos is the average value of each feature where the label is positive
        # avg_neg is the average value of each feature where the label is negative
        avg_pos = np.mean(pos_indices, axis=0)  # Eq. 3
        avg_neg = np.mean(neg_indices, axis=0)  # Eq. 4

        # weights are calculated based on Eq. 3 and Eq. 4

        self.weights = (avg_pos - avg_neg) / (avg_pos + avg_neg + 0.0000001)  # Eq. 5

        # For each record, a score is calculated. If the record is positive/negative, the score will be added to
        # posscore/negscore
        sum_scores = np.dot(X, self.weights)  # Eq. 6

        pos_label_count = np.count_nonzero(y)
        neg_label_count = y.shape[0] - pos_label_count

        # pos_score_avg and neg_score_avg are average values of records scores for positive and negative classes
        pos_score_avg = np.mean(sum_scores[y == 1])  # Eq. 7
        neg_score_avg = np.mean(sum_scores[y == 0])  # Eq. 8

        # bias is calculated using a weighted average

        self.bias = (neg_label_count * pos_score_avg + pos_label_count * neg_score_avg) / (
                neg_label_count + pos_label_count)  # Eq. 9

    def predict(self, test_data):
        """
        This is for prediction. When the model is trained, it can be applied on the test data.
        Parameters
        ----------
        test_data: either list or ndarray, two dimensional
            the data without labels in
        Returns
        ----------
        predictions in numpy array
        """
        X = test_data
        if isinstance(test_data, list):
            X = np.array(test_data, dtype="float32")

        temp = np.dot(X, self.weights)
        preds = np.where(temp <= self.bias, 0, 1)
        return preds


class SEFRMultiClass(SEFR):
    """
    This is the multiclass classifier version of the SEFR algorithm for Python
    based on https://github.com/sefr-classifier/sefr/blob/master/SEFR.py

    Also see: https://arxiv.org/abs/2006.04620
    """

    def __init__(self):
        """
        Initialize model class.
        """

        self.labels = np.array([])
        self.weights = np.array([])
        self.bias = np.array([])
        self.training_time = 0

    def fit(self, train_data, train_target):
        """
        Train the model.
        """

        self.labels = np.unique(train_target)  # get all labels
        self.weights = []
        self.bias = []
        self.training_time = 0

        start_time = time.monotonic_ns()
        train_data = np.array(train_data, dtype='float32')
        train_target = np.array(train_target, dtype='int32')

        for label in self.labels:  # train binary classifiers on each labels

            pos_labels = (train_target != label)  # use "not the label" as positive class
            neg_labels = np.invert(pos_labels)  # use the label as negative class

            pos_indices = train_data[pos_labels]
            neg_indices = train_data[neg_labels]

            avg_pos = np.mean(pos_indices, axis=0)
            avg_neg = np.mean(neg_indices, axis=0)

            weight = np.nan_to_num(
                (avg_pos - avg_neg) / (avg_pos + avg_neg))  # calculate model weight of "not the label"
            weighted_scores = np.dot(train_data, weight)

            pos_score_avg = np.mean(weighted_scores[pos_labels])
            neg_score_avg = np.mean(weighted_scores[neg_labels])

            bias = -(neg_indices.size * pos_score_avg +  # calculate weighted average of bias
                     pos_indices.size * neg_score_avg) / (neg_indices.size + pos_indices.size)

            self.weights.append(weight)  # label weight
            self.bias.append(bias)  # label bias

        self.weights = np.array(self.weights, dtype='float32')
        self.bias = np.array(self.bias, dtype='float32')
        self.training_time = time.monotonic_ns() - start_time

    def predict(self, test_data):
        """
        Predict labels of the new data.
        """

        test_data = np.array(test_data, dtype='float32')

        # calculate weighted score + bias on each labels
        weighted_score = np.add(np.dot(self.weights, test_data.T).T, self.bias)
        return self.labels[np.argmin(weighted_score, axis=1)]

    def get_params(self, deep=True):  # for cross-validation
        return {}

