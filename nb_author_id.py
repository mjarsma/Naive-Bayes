import sys;
from time import time;
from email_preprocess import preprocess;  # @UnresolvedImport
sys.path.append("/Users/Miikka/ud120-projects/tools/");

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

from sklearn.naive_bayes import GaussianNB;
### Create the classifier
clf = GaussianNB();

t0 = time();
### Fit the classifier on the training features and labels
clf.fit(features_train, labels_train);
print("training time:", round(time()-t0, 3), "s");

t0 = time();
### Use the trained classifier to predict labels for the test features
pred = clf.predict(features_test);
print("training time:", round(time()-t0, 3), "s");