### Use numpy to generate training data
import numpy as np;
features_train = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]);
labels_train = np.array([1, 1, 1, 2, 2, 2]);
features_test = np.array([[-1, -2], [-3, -1], [-2, -2], [2, 1], [2, 3], [3, 3]]);
labels_test = np.array([2, 2, 1, 1, 2, 1]);
### Import gaussian naive bayes module 
from sklearn.naive_bayes import GaussianNB;
### Create the classifier
clf = GaussianNB();
### Fit the classifier on the training features and labels
clf.fit(features_train, labels_train);
### Use the trained classifier to predict labels for the test features
pred = clf.predict(features_test);

### Calculate and return the accuracy on the test data 
from sklearn.metrics import accuracy_score;
accuracy = accuracy_score(pred, labels_test);
### Returns the mean accuracy on the given test data and labels
print(accuracy);