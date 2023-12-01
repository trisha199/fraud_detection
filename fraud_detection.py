#!/usr/bin/env python
# coding: utf-8

# In[64]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc


# In[3]:


#read transactions data
transactions = pd.read_csv('transactions_obf.csv')
transactions.head()


# In[4]:


#read fraud labels
labels = pd.read_csv('labels_obf.csv')
labels.head()


# In[5]:


#add a fraud column to the transaction data and compare it with labels data frame to match the eventiD and classify it as fraud or not
transactions['fraud'] = transactions['eventId'].isin(labels['eventId'])
#convert the fraud column to int
transactions['fraud'] = transactions['fraud'].astype(int)
transactions.head()


# In[6]:


transactions.describe()


# In[7]:


transactions.groupby('fraud').count()


# In[8]:


transactions.info()


# In[9]:


#looking for missing values
transactions.isnull().sum().sort_values()


# In[10]:


#since merchantZip has a lot of missing values and not useful for analysis, we can drop it
transactions.drop(['merchantZip'], axis=1, inplace=True)
transactions.head()


# In[11]:


#converting transactionTime to datetime
transactions['transactionTime'] = pd.to_datetime(transactions['transactionTime'])
transactions.head()


# In[12]:


#histogram of fraud and normal transactions
plt.hist(transactions['fraud'])
plt.title('Histogram of Fraud and Normal Transactions')
plt.xlabel('Fraud and Normal Transactions')
plt.ylabel('Frequency')
plt.show()


# In[13]:


# Filter transactions with fraud == 1 and plot the average transaction amount for each available cash amount
fraudulent_transactions = transactions[transactions['fraud'] == 1]
fraudulent_transactions['transactionAmount'].groupby(fraudulent_transactions['availableCash']).mean().plot(kind='bar')


# In[14]:


# Filter only the fraudulent transactions and then group by merchantId and count
fraud_transactions = transactions[transactions['fraud'] == 1].groupby('merchantId').size()

# Plot the top 10 merchants with the highest count of fraudulent transactions
fraud_transactions.sort_values(ascending=False).head(10).plot(kind='barh', figsize=(10,5), title='Number of fraudulent transactions by merchant ID')


# In[15]:


#check for any outliers in the data
sns.boxplot(x=transactions['transactionAmount'])


# In[16]:


transactions.columns


# In[17]:


# Filter only the fraudulent transactions and then group by accountNumber and count
fraud_transactions = transactions[transactions['fraud'] == 1].groupby('accountNumber').size()

# Plot the top 10 accounts with the highest count of fraudulent transactions
fraud_transactions.sort_values(ascending=False).head(10).plot(kind='bar', figsize=(10,5), title='Number of fraudulent transactions by Account Number')


# In[18]:


# Filter only the fraudulent transactions and then group by posEntryMode and count
fraud_transactions = transactions[transactions['fraud'] == 1].groupby('posEntryMode').size()

# Select the top 10 POS Entry Modes with the highest count of fraudulent transactions
top_fraud_posEntryModes = fraud_transactions.nlargest(10)

# Create a pie chart
fig, ax = plt.subplots()
ax.pie(top_fraud_posEntryModes, labels = top_fraud_posEntryModes.index, autopct='%1.1f%%')
plt.title('Proportion of fraudulent transactions by POS Entry Mode')
plt.show()


# Looks like POS 81 entry style has the most fradulent transactions

# In[19]:


#modify the TransactionTime column to extract the day of the week and the hour of the day
transactions['DayOfWeek'] = transactions['transactionTime'].dt.day_name()
transactions['HourOfDay'] = transactions['transactionTime'].dt.hour


# In[20]:


#check if transaction fraud is higher any specific day of the week
fraud_transactions = transactions[transactions['fraud'] == 1].groupby('DayOfWeek')['fraud'].count()
fraud_transactions.plot(kind='bar', figsize=(10,5), title='Fraud Transactions by Day of Week')


# In[21]:


#check if transaction fraud is higher any specific time of the day
fraud_transactions = transactions[transactions['fraud'] == 1].groupby('HourOfDay')['fraud'].count()
fraud_transactions.plot(kind='bar', figsize=(10,5), title='Fraud Transactions by Hour of Day')


# In[22]:


transactions.head()


# In[23]:


#dropping eventID and traactionTime since we have fraud as a unique identifier for each event
transactions = transactions.drop(['eventId','transactionTime'], axis=1)


# In[24]:


transactions.info()


# In[25]:


#Need to change the label encoding for the categorical variables to numerical variables for accountNumber, MerchantID, DayOfWeek
le = LabelEncoder()
transactions['accountNumber'] = le.fit_transform(transactions['accountNumber'])
transactions['merchantId'] = le.fit_transform(transactions['merchantId'])
transactions['DayOfWeek'] = le.fit_transform(transactions['DayOfWeek'])
transactions.info()


# In[26]:


#Getting the correlation between the columns
transactions.corr()


# In[27]:


#creating a heatmap to see the correlation between the features spearman method
plt.figure(figsize=(20,10))
sns.heatmap(transactions.corr(), annot=True, cmap='coolwarm')


# In[28]:


#trying corelation matrix for kendal to see if there is any non-linear corelation between the variables
transactions.corr(method='kendall')

#plotting the corelation matrix
plt.figure(figsize=(15,8))
sns.heatmap(transactions.corr(method='kendall'), annot=True, cmap='coolwarm')


# In[29]:


#split the data into X(features) and y(target)
X = transactions.drop('fraud', axis=1).values
y = transactions['fraud'].values


# In[30]:


#scaling the data to bring all the features to the same level of magnitude
scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[31]:


#splitting the data into training and testing data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state = 42)


# In[32]:


#Creating a KNN model to predict fraud
knn = KNeighborsClassifier(n_neighbors=5)
#fit the classifier to the data
knn.fit(X_train,y_train)
#predict the response
y_pred = knn.predict(X_test)
#Getting the accuracy of the model
knn.score(X_test,y_test)


# In[34]:


# Creating neighbors and accuracies lists for neighbors between 1 and 13
neighbors = np.arange(1, 13)
train_accuracies = {}
test_accuracies = {}

for neighbor in neighbors:
  
	# Setting up a KNN Classifier
	knn = KNeighborsClassifier(n_neighbors=neighbor)
  
	# Fitting the model
	knn.fit(X_train, y_train)
  
	# Compute accuracy
	train_accuracies[neighbor] = knn.score(X_train, y_train)
	test_accuracies[neighbor] = knn.score(X_test, y_test)
print(neighbors, '\n', train_accuracies, '\n', test_accuracies)


# In[35]:


#visualizing model complexity
plt.title("KNN: Varying Number of Neighbors")

# Plotting training accuracies
plt.plot( neighbors,train_accuracies.values(), label="Training Accuracy")

# Plotting test accuracies
plt.plot( neighbors,test_accuracies.values(), label="Testing Accuracy")

plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")

# Display the plot
plt.show()


# In[36]:


#based on this model, looks like the best neighbors is 8
#Creating a KNN model to predict fraud
knn = KNeighborsClassifier(n_neighbors=8)
#fitting the classifier to the data
knn.fit(X_train,y_train)
#predicting the response
y_pred = knn.predict(X_test)
#Getting the accuracy of the model
knn.score(X_test,y_test)


# In[37]:


# Import confusion matrix
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Generate the confusion matrix and classification report and roc_auc_score
print(roc_auc_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[38]:


# Predict probabilities
y_pred_probs = knn.predict_proba(X_test)[:,1]

# Import roc_curve
from sklearn.metrics import roc_curve

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)

plt.plot([0, 1], [0, 1], 'k--')

# Plot tpr against fpr
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Fraud Prediction')
plt.show()


# This Curve doesn't look very good. We will try Logistic Regression and see that improves

# In[39]:


#let's try logistic regression
from sklearn.linear_model import LogisticRegression

# Create an instance of the model
logreg = LogisticRegression()

# Fit the model
logreg.fit(X_train,y_train)

# Predict probabilities
y_pred_probs = logreg.predict_proba(X_test)[:,1]
print(y_pred_probs[:10])


# In[40]:


# Import roc_curve
from sklearn.metrics import roc_curve

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)

plt.plot([0, 1], [0, 1], 'k--')

# Plot tpr against fpr
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Fraud Prediction')
plt.show()


# In[41]:


# Calculate roc_auc_score
print(roc_auc_score(y_test, y_pred_probs))

# Calculate the confusion matrix
print(confusion_matrix(y_test, y_pred))

# Calculate the classification report
print(classification_report(y_test, y_pred))


# In[42]:


#trying cross validation to see if it improves the model
from sklearn.model_selection import KFold, cross_val_score
# Create a KFold object
kf = KFold(n_splits=6, shuffle=True, random_state=5)

reg = LogisticRegression()

# Compute 6-fold cross-validation scores
cv_scores = cross_val_score(reg, X, y, cv=kf)

# Print scores
print(cv_scores)


# In[43]:


# Getting the mean CV score
print(np.mean(cv_scores))

# Getting the std CV score
print(np.std(cv_scores))

# Print the 95% confidence interval
print(np.quantile(cv_scores, [0.025, 0.975]))


# In[44]:


#Performing hyperparameter tuning using GridSearchCV
# Import RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Set up the parameter grid
params = {"penalty": ["l2"],
         "tol": np.linspace(0.0001, 1.0, 50),
         "C": np.linspace(0.1, 1, 50),
         "class_weight": ["balanced", {0:0.8, 1:0.2}]}


# Instantiate lasso_cv
logreg_cv = RandomizedSearchCV(logreg, params, cv=kf)

# Fit to the training data
logreg_cv.fit(X_train,y_train)
print("Tuned logreg paramaters: {}".format(logreg_cv.best_params_))
print("Tuned logreg score: {}".format(logreg_cv.best_score_))


# In[45]:


#trying the logistic regression model with the hyperparameters found in the grid search

logreg = LogisticRegression(solver='lbfgs', penalty='l2', C=0.15510204081632656, class_weight={0: 0.8, 1: 0.2})

# Fit the model.
logreg.fit(X_train, y_train)

# Predict on the test set.
y_pred = logreg.predict(X_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# In[46]:


# Generate the confusion matrix and classification report and roc_auc_score
print(roc_auc_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Even after the hyperparameter tuning, this model doesn't yield better results. 

# Checking if other models might be better and trying crossvalidation

# In[57]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# Create models dictionary
models = {"Logistic Regression": LogisticRegression(), "KNN": KNeighborsClassifier(), "Decision Tree ": DecisionTreeClassifier(), "Random Forest ": RandomForestClassifier()}
results = []

# Loop through the models' values
for model in models.values():
  
  # Instantiate a KFold object
  kf = KFold(n_splits=6, random_state=12, shuffle=True)
  
  # Perform cross-validation
  cv_results = cross_val_score(model, X_train, y_train, cv=kf)
  results.append(cv_results)
plt.boxplot(results, labels=models.keys())
plt.title("Model Comparison")
plt.figure(figsize=(15,10))
plt.show()


# Looks Like might perform better KNN and Random Forest perform better with CrossValidation 

# In[62]:


# Import confusion matrix
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

knn = KNeighborsClassifier(n_neighbors=6)

# Fit the model to the training data
knn.fit(X_train, y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report and roc_auc_score
print(roc_auc_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[68]:


# Import roc_curve
from sklearn.metrics import roc_curve,auc

y_pred_prob = knn.predict_proba(X_test)[:,1]
# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)

# Compute the area under the ROC curve
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()


# Let's see if CrossValidation improves this score

# In[69]:


from sklearn.model_selection import cross_val_score

# Create a new KNN model
knn_cv = KNeighborsClassifier(n_neighbors=6)

# train model with cv of 5 
cv_scores = cross_val_score(knn_cv, X_train, y_train, cv=5)

# print each cv score (accuracy) and average them
print(cv_scores)
print(f'cv_scores mean:{np.mean(cv_scores)}')


# In[70]:


#check to see if the model has improved
knn_cv.fit(X_train, y_train)
y_pred = knn_cv.predict(X_test)
print(roc_auc_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[72]:


# Import roc_curve
from sklearn.metrics import roc_curve,auc

y_pred_prob = knn_cv.predict_proba(X_test)[:,1]
# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)

# Compute the area under the ROC curve
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()


# In[73]:


print(roc_auc_score(y_test, y_pred))
print(auc(fpr, tpr))


# There is no change in model performance. Let try Random Forest and since crossvalidation has no effect, we will not be doing it for RandomForest

# In[74]:


# Import the necessary libraries
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier
rf = RandomForestClassifier(n_estimators=40, class_weight='balanced', random_state=42)

# Fit the model to the training data
rf.fit(X_train, y_train)

# Predict the labels of the test data: y_pred
y_pred = rf.predict(X_test)

# Generate the confusion matrix, classification report, and roc_auc_score
print(roc_auc_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# The F1 and Recall scores have improved indicating that fraud detection number has gone up significantly but let's check the ROC curves.

# In[75]:


# Import roc_curve
from sklearn.metrics import roc_curve,auc

y_pred_prob = rf.predict_proba(X_test)[:,1]
# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)

# Compute the area under the ROC curve
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()


# The ROC curve hasn't improved much. This is might be due to class Imbalace. We can do oversampling using SMOTE to see we can improve this.

# In[76]:


from imblearn.over_sampling import SMOTE
# Assume X_train and y_train are your data
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)


# In[77]:


# Import the necessary libraries
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier
rf = RandomForestClassifier(n_estimators=40, class_weight='balanced', random_state=42)

# Fit the model to the training data
rf.fit(X_res, y_res)

# Predict the labels of the test data: y_pred
y_pred = rf.predict(X_test)

# Generate the confusion matrix, classification report, and roc_auc_score
print(roc_auc_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[78]:


# Import necessary libraries
from sklearn.metrics import roc_curve, auc

#Getting the prediction probabilities using the random forest classifier
y_pred_prob = rf.predict_proba(X_test)[:,1]

# Getting the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

#Getting the area under the ROC curve
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()


# In[79]:


print(roc_auc_score(y_test, y_pred))
print(auc(fpr, tpr))


# The AOC and the fraud detection has improved significantly due to oversampling. Let's try to do some randomized grid search to see if we can get better hyperparameters to try

# In[82]:


from sklearn.model_selection import RandomizedSearchCV
#silence warnings
import warnings
warnings.simplefilter('ignore', category=FutureWarning)
warnings.filterwarnings('ignore')


# Define the parameter grid
param_grid = {
    'n_estimators': [30, 50, 100, 200],
    'max_features': ['sqrt'],
    'max_depth': [10, 20, 30, 40, 50, None],
    'class_weight': ['balanced', None],
    'criterion': ['gini', 'entropy']
}

# Initialize a Random Forest classifier
rf = RandomForestClassifier(random_state=42)

# Initialize GridSearchCV
grid_search = RandomizedSearchCV(estimator=rf,cv=5,param_distributions=param_grid, scoring='roc_auc', n_jobs=-1, verbose=1)

# Fit GridSearchCV to the resampled data
grid_search.fit(X_res, y_res)

# Get the best parameters
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# Fit the model to the training data using best parameters
rf_best = RandomForestClassifier(**best_params, random_state=42)
rf_best.fit(X_res, y_res)

# Predict the labels of the test data: y_pred
y_pred = rf_best.predict(X_test)

# Generate the confusion matrix, classification report, and roc_auc_score
print(roc_auc_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# The model has improved significantly with a recall score of 0.59. 

# In[83]:


# Import necessary libraries
from sklearn.metrics import roc_curve, auc

#Getting the prediction probabilities using the random forest classifier
y_pred_prob = rf_best.predict_proba(X_test)[:,1]

# Getting the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

#Getting the area under the ROC curve
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()


# In[84]:


print(roc_auc_score(y_test, y_pred))
print(auc(fpr, tpr))


# 
