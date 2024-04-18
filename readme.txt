Prof advice:

Make our own data set: do ~610 fraud and and add 600 legit emails from Zenodo data and add 10 emails from our own inboxes that are legit
find P, A, R, F for if we ONLY use KNN, for if we ONLY use BT, for if we use ensemble for BOTH.
Create table/bargraph to show those three ^ as well as the mean/avg of them
Train and find (separetly) for both the navario_5.csv and dataset we created
Make bar graphs, one for Navario_5 and one for self created dataset (graph to show precision, acuracy, F and R)


Dataset: https://zenodo.org/records/8339691

Code above: https://www.datacamp.com/tutorial/decision-tree-classification-python

Loads libraries
Reads dataset
Labels columns within data
Tested - data print out

Nazario data set:

Contains ~3000 emails: 300 legit & 2,700 fraud //NOT EQUAL FIX LATER

Contains the following catagories: 'sender', 'receiver', 'date', 'subject', 'body', 'url'

This code uses the train_test_split function from the sklearn.model_selection module to split a dataset into a training set and a test set.
The X and y variables represent the features and target variable of the dataset, respectively.
The test_size parameter is set to 0.3, which means that 30% of the data will be used for testing and 70% will be used for training.
The random_state parameter is set to 1, which ensures that the same random split is generated each time the code is run.
The function returns four arrays: X_train, X_test, y_train, and y_test.
These arrays can be used to train and evaluate a machine learning model.

Creates a decision tree classifier object using the DecisionTreeClassifier() function.
Then, it trains the classifier using the fit() method with the training data X_train and y_train.
Finally, it uses the trained classifier to predict the response for the test dataset X_test and stores the predictions in y_pred.
