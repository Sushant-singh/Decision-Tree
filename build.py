# NAME - SUSHANT SINGH
# Decision Tree Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import tree

# Importing the dataset
dataset = pd.read_csv('Iris.csv')

# Printing the First 5 rows
dataset.head()

# Decribing the data in the Columns
dataset.describe()

# Checking if there are any Null Values
dataset.isnull().any()

# Extracting the Independent variable X and the Dependent variable y
X = dataset.iloc[:, 1:5].values
y = dataset.iloc[:, -1].values

# Creating a temp dataset without the ID column to visualize as a box plot
temp = dataset
temp.drop("Id", axis = 1, inplace = True)
temp.boxplot()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Decision Tree Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualizing the Confusion Matrix as a heat map
sns.heatmap(cm, annot=True)

# Printing accuracy of the Model
print("Accuracy of this model is" , classifier.score(X_test, y_test))

# Visualizing the Decision Tree
fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn=['setosa', 'versicolor', 'virginica']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(classifier,
               feature_names = fn, 
               class_names=cn,
               filled = True);
               