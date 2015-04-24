mport pandas as pd

df = pd.read_csv("/home/vagrant/repos/datasets/train.csv")

df.describe()

--

import re

target_data = df["label"]
feature_data = df[df.columns[1:]]

from sklearn.cross_validation import train_test_split
full_X_train, full_X_test, full_y_train, full_y_test = train_test_split(
    feature_data, target_data, test_size=0.3)

X_train = full_X_train[:len(full_X_train)/8]
y_train = full_y_train[:len(full_y_train)/8]
X_test = full_X_test[:len(full_X_test)/8]
y_test = full_y_test[:len(full_y_test)/8]

--

import time
from sklearn.metrics import classification_report, accuracy_score
from nolearn.dbn import DBN

start = time.time()
print "\nDEEP BELIEF NETWORK\n"
dbn_clf = DBN(
    [X_train.shape[1], 40, 10],
    learn_rates=0.1,
    epochs=4)

# Train the DBN
dbn_clf.fit(X_train, y_train)

# Make predictions and test the accuracy
y_pred = dbn_clf.predict(X_test)
print "Accuracy: {}".format(accuracy_score(y_pred, y_test))

end = time.time()
print "Took {} s to run.".format(end - start)

--

target_data = df["label"]
feature_data = df[df.columns[1:]]

# Scale each feature to 0-1 scale
feature_data = feature_data / 255.0

from sklearn.cross_validation import train_test_split
full_X_train, full_X_test, full_y_train, full_y_test = train_test_split(
    feature_data, target_data, test_size=0.3)

X_train = full_X_train[:len(full_X_train)/8]
y_train = full_y_train[:len(full_y_train)/8]
X_test = full_X_test[:len(full_X_test)/8]
y_test = full_y_test[:len(full_y_test)/8]

--

start = time.time()
print "\nDEEP BELIEF NETWORK\n"

accuracies=[]
for rate in (.1, .001, .3):
   dbn_clf = DBN(
       [X_train.shape[1], 500, 10],
       learn_rates=rate,
       epochs=10,
       verbose=1)

   # Train the DBN
   dbn_clf.fit(X_train, y_train)

   # Make predictions and test the accuracy
   y_pred = dbn_clf.predict(X_test)

print "Accuracy: {}".format(accuracy_score(y_pred, y_test))

end = time.time()
print "Took {} s to run.".format(end - start)

--


