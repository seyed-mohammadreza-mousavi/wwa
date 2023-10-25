import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.datasets import make_gaussian_quantiles
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import seaborn as sns
import random
from sklearn.metrics import classification_report, roc_auc_score

# Generate a binary dataset
#X, y = make_classification(n_samples=10000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)
#X, y = make_blobs(n_samples=10000, n_features=2, centers=[(-2, -2), (2, 2)], cluster_std=0.5, random_state=42)
#X, y = make_blobs(n_samples=10000, n_features=2, centers=[(-2, -2), (2, 2)], cluster_std=[1.5, 1.5], random_state=42)

# Set the parameters for the imbalanced dataset
n_samples = 10000
n_features = 2
weights = [0.97, 0.03]  # Adjust the class weights here
class_sep = 2.0

# Set the mean and covariance for the minority class
mu_minority = np.array([1, 1])  # Mean of the minority class
cov_minority = np.array([[1, 0], [0, 1]])  # Covariance matrix of the minority class

# Set the mean and covariance for the majority class
mu_majority = np.array([-1, -1])  # Mean of the majority class
cov_majority = np.array([[1, 0], [0, 1]])  # Covariance matrix of the majority class

# Generate samples for the minority class
minority_samples = int(n_samples * weights[1])
X_minority = np.random.multivariate_normal(mu_minority, cov_minority, minority_samples)
y_minority = np.ones(minority_samples, dtype=int)

# Generate samples for the majority class
majority_samples = n_samples - minority_samples
X_majority = np.random.multivariate_normal(mu_majority, cov_majority, majority_samples)
y_majority = np.zeros(majority_samples, dtype=int)

# Concatenate the minority and majority class samples
X = np.concatenate((X_majority, X_minority))
y = np.concatenate((y_majority, y_minority))

# Separate features
feature_1 = X[:, 0]
feature_2 = X[:, 1]
feature_1.shape, feature_2.shape

# Visualize the dataset
plt.scatter(feature_1, feature_2, c=y)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Binary Dataset')

# Add legend labels
plt.legend(['Class 0', 'Class 1'])
plt.xlim([-5, 5])

plt.show()

# Oversample the minority class using a chaotic map
chaotic_map = lambda x: 3.9 * x * (1 - x)
n_oversample = majority_samples - minority_samples

np.random.seed(42)  # Set a seed for reproducibility
x0 = np.random.rand()
x = x0
oversampled_X_minority = np.zeros((n_oversample, n_features))
for i in range(n_oversample):
    x = chaotic_map(x)
    oversampled_X_minority[i] = mu_minority + x * (cov_minority @ np.random.randn(n_features))

# Concatenate the minority and majority class samples
X = np.concatenate((X_majority, X_minority, oversampled_X_minority))
y = np.concatenate((y_majority, y_minority, np.ones(n_oversample, dtype=int)))

X.shape, y.shape, len(X_minority), len(X_majority)

# Separate features
feature_1 = X[:, 0]
feature_2 = X[:, 1]
feature_1.shape, feature_2.shape

# Visualize the dataset
plt.scatter(feature_1, feature_2, c=y)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Binary Dataset on oversampled minority class')

# Add legend labels
plt.legend(['Class 0', 'Class 1'])
plt.xlim([-5, 5])

plt.show()

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set the number of epochs
max_epochs = 2000
stopper=500

# Initialize the classifier
#classifier = SVC()
#classifier = SGDClassifier(loss='hinge')
classifier = LogisticRegression()

# Initialize variables to track the best F-measure and corresponding epoch
best_f_measure = 0
best_epoch = 0

for epoch in range(1, max_epochs + 1):
    # Shuffle the training data
    #random.shuffle(X_train)
    #random.shuffle(y_train)
    
    # Train the classifier
    classifier.fit(X_train, y_train)
    #classifier.partial_fit(X_train, y_train, classes=np.unique(y_train))
    
    # Make predictions on the test set
    y_pred = classifier.predict(X_test)
    
    # Calculate F-measure
    f_measure = f1_score(y_test, y_pred)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print the results for the current epoch
    print(f"Epoch {epoch}:")
    print("F-measure:", f_measure)
    print("Confusion Matrix:")
    print(cm)
    print("Accuracy:", accuracy)
    print()
    
    # Check if the current F-measure is the best so far
    if f_measure > best_f_measure:
        best_f_measure = f_measure
        best_epoch = epoch
    
    # Check if the F-measure has not improved for a certain number of epochs
    if epoch - best_epoch >= stopper:
        print(f"F-measure has not improved for {stopper} epochs. Stopping training.")
        break

# Train a classifier (Support Vector Machine in this example)
#classifier = SVC()
#classifier.fit(X_train, y_train)

# Make predictions on the test set
#y_pred = classifier.predict(X_test)

# Calculate F1 score
f1 = f1_score(y_test, y_pred)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("F1 Score:", f1)
print("Confusion Matrix:")
print(cm)
print("True Positive (TP):", tp)
print("True Negative (TN):", tn)
print("False Positive (FP):", fp)
print("False Negative (FN):", fn)
# Visualize the confusion matrix
labels = ['Class 0', 'Class 1']
sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Calculate the total number of instances
total_instances = len(y_test)

# Calculate the percentages
tp_percentage = tp / total_instances * 100
tn_percentage = tn / total_instances * 100
fp_percentage = fp / total_instances * 100
fn_percentage = fn / total_instances * 100

print("Confusion Matrix:")
print(cm)
print("True Positive (TP):", tp, "({:.2f}%)".format(tp_percentage))
print("True Negative (TN):", tn, "({:.2f}%)".format(tn_percentage))
print("False Positive (FP):", fp, "({:.2f}%)".format(fp_percentage))
print("False Negative (FN):", fn, "({:.2f}%)".format(fn_percentage))

# Visualize the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(feature_1, feature_2, c=y, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Binary Dataset with Decision Boundary')
plt.legend(['Class 0', 'Class 1'])
plt.show()