mport pandas as pd
from sklearn import feature_extraction
from sklearn.cross_validation import cross_val_score
from sklearn import tree
from sklearn import ensemble
from sklearn import neighbors
from sklearn import linear_model
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import statsmodels.api as sm

df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data", header=None)
df = df.interpolate()
# Uh oh its a string, so let's convert that data in column 1 to numerical data
df = df.convert_objects(convert_numeric=True)

# Let's go ahead and remove the rows that have missing values in the categorical columns
# since we have no natural imputation we want to apply for missing categorical data
categorical_columns = [0, 3, 4, 5, 6, 8, 9, 11, 12, 15]
for i in categorical_columns:
    df = df[df[i] != '?']

# DictVectorizer is handy tool that converts non-ordinal categorical columns
# into numerical columns, but it will introduce multicollinearity

# Documentation here:
# http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/preprocessing/feature_encoding.ipynb
dvec = feature_extraction.DictVectorizer(sparse=False)
X = dvec.fit_transform(df.transpose().to_dict().values())

# the -1th column is whether not the row was tagged as '-'
X[:,-1]

# the -2th column is whether not the row was tagged as '+'
X[:,-2]

# Since these are collinear columns and just opposites of each other, we don't need to consider them both

feature_data = X[:,:-2]
target = X[:,-2]

feature_data.shape

target.shape

# Oh no, we have some NaN values in some spots!
# Let's drop the rows that contain any NaN values
df = df.dropna()

# Let's DictVectorize the dataframe again and separate
# features and target
X = dvec.fit_transform(df.transpose().to_dict().values())
feature_data = X[:,:-2]
target = X[:,-2]

# looks like we've lost 18 rows, but that's okay
print feature_data.shape
print target.shape

print "\nDECISION TREE\n"
decision_tree_clf = tree.DecisionTreeClassifier()
scores = cross_val_score(decision_tree_clf, feature_data, target, cv=3)
print "Mean: {}".format(scores.mean())
print "Std Dev: {}".format(np.std(scores))

print "\nRANDOM FOREST\n"
# see what happens as we bump up the number of estimators
random_forest_clf = ensemble.RandomForestClassifier(n_estimators=20)
scores = cross_val_score(random_forest_clf, feature_data, target, cv=3)
print "Mean: {}".format(scores.mean())
print "Std Dev: {}".format(np.std(scores))

print "\nBOOSTED TREES\n"
boosted_clf = ensemble.AdaBoostClassifier(n_estimators=20)
scores = cross_val_score(boosted_clf, feature_data, target, cv=3)
print "Mean: {}".format(scores.mean())
print "Std Dev: {}".format(np.std(scores))

print "\nkNN\n"
knn_clf = neighbors.KNeighborsClassifier(n_neighbors=7)
scores = cross_val_score(knn_clf, feature_data, target, cv=3)
print "Mean: {}".format(scores.mean())
print "Std Dev: {}".format(np.std(scores))

print "\nLogistic\n"
logistic_clf = linear_model.LogisticRegression()
scores = cross_val_score(logistic_clf, feature_data, target, cv=3)
print "Mean: {}".format(scores.mean())
print "Std Dev: {}".format(np.std(scores))

print "\nSVC\n"
# C is a tuning a parameter (remember it's our error budget for how much slack we give the hyperplane)
# mess around with C and see what happens
support_vector_clf = svm.LinearSVC(C=50)
scores = cross_val_score(support_vector_clf, feature_data, target, cv=3)
print "Mean: {}".format(scores.mean())

# code that get covariance matrix (a matrix that sees how) correlated two features are.
# with second line we can see that eigenvalues of the eigenvalues of the columns 
# if the eigenvalues of say two columns are 0 

# could the data be overfitting
# Now we want to train on 20% of the data and work on 80% of the data.  By definition this can't be overfitting.  
# Oh no, we have some NaN values in some spots!
# Let's drop the rows that contain any NaN values
df = df.dropna()

# Let's DictVectorize the dataframe again and separate
# features and target
X = dvec.fit_transform(df.transpose().to_dict().values())
feature_data = X[:,:-2]
target = X[:,-2]

# looks like we've lost 18 rows, but that's okay
print feature_data.shape
print target.shape

--

feature_data[0]

--

df.head(1)

--

pd.DataFrame(X, columns=dvec.get_feature_names()).head()

--

# Ok looks fie, but what about multicollinearity? 
# Shouldn't logistic regression fail since it won't be able to invert the matrix?
# Let's find out!

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

print "\nLogistic\n"
logistic_clf = linear_model.LogisticRegression(penalty='l2')
a_train, a_test, b_train, b_test = train_test_split(feature_data, target, test_size=0.80, random_state=42)
logistic_clf.fit(a_train, b_train)
print accuracy_score(logistic_clf.predict(a_test), b_test)
logistic_clf.coef_

--

# Let's see what the multicollinearity situation here is
# Documentation here: http://stackoverflow.com/a/25833792
import numpy as np
corr = np.corrcoef(feature_data, rowvar=0)
w, v = np.linalg.eig(corr)

--

# Looks like there's some weird j term
print w

--

type(w[0])
# Let's print this out more human readably
np.set_printoptions(suppress=True)


--

# Let's convert to real numbers
w_real = np.real_if_close(w)
print w_real

--

# Let's convert to real numbers
w_real = np.real_if_close(w)
print w_real

--

# what does w_real correspond to? one value for each column
len(w_real)

--

# Let's try to invert the matrix
np.linalg.inv(corr)

--

 Wow a lot of 0s, means multicolinearity between some of the variables
# We expected this right?
# So why does Logistic Regression still work if it can't invert the matrix?
# Just to check against another library, let's try statsmodels
import statsmodels.api as sm

print "\nLogistic with Statsmodels\n"
a_train, a_test, b_train, b_test = train_test_split(feature_data, target, test_size=0.80, random_state=42)
logit = sm.Logit(b_train, a_train)
result = logit.fit()
result.summary()

print accuracy_score(logit.predict(a_test), b_test)

--

# Wow a lot of 0s, means multicolinearity between some of the variables
# We expected this right?
# So why does Logistic Regression still work if it can't invert the matrix?
# Just to check against another library, let's try statsmodels
import statsmodels.api as sm

print "\nLogistic with Statsmodels\n"
a_train, a_test, b_train, b_test = train_test_split(feature_data, target, test_size=0.80, random_state=42)
logit = sm.Logit(b_train, a_train)
result = logit.fit()
result.summary()

print accuracy_score(logit.predict(a_test), b_test)

--


for 
print "\nSVC\n"
# C is a tuning a parameter (remember it's our error budget for how much slack we give the hyperplane)
# mess around with C and see what happens
support_vector_clf = svm.LinearSVC(C=100)
scores = cross_val_score(support_vector_clf, feature_data, target, cv=5)
print "Mean: {}".format(scores.mean())
# Mean is basically is the accuracy.
print "Std Dev: {}".format(np.std(scores))

--

print "\nRANDOM FOREST\n"
# see what happens as we bump up the number of estimators
random_forest_clf = ensemble.RandomForestClassifier(n_estimators=20)
scores = cross_val_score(random_forest_clf, feature_data, target, cv=5)
print "Mean: {}".format(scores.mean())
print "Std Dev: {}".format(np.std(scores))

--

import matplotlib.pyplot as plt


x = range(1, 100)
y = []
for k in x:
# C is a tuning a parameter (remember it's our error budget for how much slack we give the hyperplane)
   support_vector_clf = svm.LinearSVC(C=k)
   scores = cross_val_score(support_vector_clf, feature_data, target, cv=5)
   y.append(scores.mean())

plt.plot(x, y)

--

dir(support_vector_clf)

--

:wq

