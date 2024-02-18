###########################################################
##        1. Naive Bayes                                 ##
##        2. Prediction                                  ##
##        3. evaluation                                  ##
##        4. probability of each class.                  ##
##        5. class labels known to the classifier.       ##
###########################################################
import numpy as np
import math
import pandas as pd
from sklearn.model_selection import train_test_split


class NBGaussian:
    def __init__(self):

        self.data = None  # X
        self.target = None  # y
        self.target_class = None
        self.columns = None  # List of features

        self.det = []  # Calculating Standard deviation list
        self.const = []
        self.stdev_list = []  # Standard deviation list
        self.mean = []  # Mean of each feature

        self.classes_ = []  # List of classes(labels)
        self.class_prior_ = None  # Probability for each classes(labels)

    def train_data(self, X_data, y_data, target_column):

        self.target = y_data
        self.target_class = target_column
        self.data = X_data
        self.columns = X.columns
        self.process()

    # Calculate the Likelihood
    def process(self):

        # Finding classes in data
        self.finding_classes()
        # Calculating probability for each classes
        self.class_prior_ = [list(self.target).count(i) / len(self.target) for i in self.classes_]
        # Splitting the data by classes
        data_split = self.splitting_data()
        # Calculating Standard deviation for each classes
        self.calculating_determinant(data_split)

        for i in data_split:
            # Finding mean of each feature
            self.mean.append(np.array(i.mean()).reshape(-1, 1))

        # Calculating const for each classes
        self.const = [1 / ((2 * math.pi) ** 0.5 * j) for j in self.det]

    def finding_classes(self):

        for i in self.target:
            if i not in self.classes_:
                self.classes_.append(i)
        self.classes_.sort()


    def splitting_data(self):

        df = pd.merge(self.data, self.target, left_index=True, right_index=True)
        data_split = []

        for i in self.classes_:

            data_class = [j for j in df.iloc if j[self.target_class] == i]

            data_class = pd.DataFrame(data_class, columns=self.columns)
            data_split.append(data_class)
        return data_split

    def calculating_determinant(self, data_split):

        for i in data_split:

            local_stdev_list = i.std().values.reshape(-1, 1)
            self.stdev_list.append(local_stdev_list)
            local_det = 1

            for j in local_stdev_list:
                for z in j:
                    local_det = local_det * z
            self.det.append(local_det)

    # Predicting test data
    def predict(self, test_data):
        predict_list = []

        for i in test_data.iloc:
            i = np.array(i).reshape(-1, 1)

            maximum = float("-inf")  # The maximum probability value
            maximum_index = -1  # The index maximum probability value
            for j in range(len(self.classes_)):

                for z in range(len(i)):
                    diff = i - self.mean[j]  # The difference between sample and the mean
                    e_part = (-0.5 * (diff ** 2 / (self.stdev_list[j]) ** 2))

                score = self.const[j] * math.e ** sum(e_part)

                if maximum < score:
                    maximum, maximum_index = score, j
            predict_list.append(maximum_index)
        return predict_list

    # evaluating the performance of NBGaussian model.
    @staticmethod
    def score(predict_list, true_list):
        zip_list = zip(predict_list, true_list)
        true_number, false_number = 0, 0

        for i in zip_list:
            if i[0] == i[1]:
                true_number += 1
            else:
                false_number += 1
        return true_number / len(true_list)


################################################TEST################################################
data = pd.read_csv("data.csv")
X = data.drop("Purchased", axis=1)
y = data["Purchased"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = NBGaussian()
model.train_data(X_train, y_train, "Purchased")
y_pred = model.predict(X_test)
point = model.score(y_pred, y_test)
print(point)
