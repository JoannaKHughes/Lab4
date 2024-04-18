# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.neighbors import KNeighborsClassifier # Import Knearest classifier
from sklearn.linear_model import LogisticRegression # Import Logistic regression for ensemble
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.ensemble import StackingClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from google.colab import files

col_names = ['sender', 'receiver', 'date', 'subject', 'body', 'url', 'label']
# load dataset
data = pd.read_csv("/content/Nazario_5.csv", header=None, names=col_names)

# Assume that the first column is the feature and the second column is the target
X = data.iloc[:, :-1] 
y = data.iloc[:, -1]

###From Geeksforgeeks
# computing number of rows
rows = len(data.axes[0])
 
# computing number of columns
cols = len(data.axes[1])
 
#print("Number of Rows: ", rows)
#print("Number of Columns: ", cols)

#print the first few
print(data.head());



# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test


# Create Decision Tree classifer object
dt = DecisionTreeClassifier()

# Creat K nearest classifier
knn = KNeighborsClassifier()

# Train Decision Tree Classifer
dt = dt.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = dt.predict(X_test)


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# Create a stacking ensemble
stack = StackingClassifier(estimators=[('dt', dt), ('knn', knn)], final_estimator=LogisticRegression())

# Fit ensemble on data
stack.fit()

# Make predictions
y_pred = stack.predict(X)

# Evaluate performance
acc = accuracy_score(y, y_pred)
acc = acc*100
print(f'Accuracy: {acc:.2f} %')
