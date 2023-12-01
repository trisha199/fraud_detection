# script.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, auc
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_curve
import seaborn as sns
import warnings
warnings.simplefilter('ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

def load_data(file_path1, file_path2):
    # Function to load data
    transactions = pd.read_csv(file_path1)
    labels = pd.read_csv(file_path2)
    return transactions, labels

def get_features_and_target(transactions, labels):
    # Function to get features and target
    transactions['fraud'] = transactions['eventId'].isin(labels['eventId'])
    #convert the fraud column to int
    transactions['fraud'] = transactions['fraud'].astype(int)
    return transactions

def preprocess(transactions):
    #Convert the transactionTime column to datetime
    transactions['transactionTime'] = pd.to_datetime(transactions['transactionTime'])
    #modify the TransactionTime column to extract the day of the week and the hour of the day
    transactions['DayOfWeek'] = transactions['transactionTime'].dt.day_name()
    transactions['HourOfDay'] = transactions['transactionTime'].dt.hour
    #dropping eventID and transactionTime since we have fraud as a unique identifier for each event
    transactions = transactions.drop(['eventId','transactionTime','merchantZip'], axis=1)
    return transactions

def plot_fraud(transactions):
    # Function to plot fraud by hour
    #check if transaction fraud is higher any specific day of the week
    fraud_transactions = transactions[transactions['fraud'] == 1].groupby('DayOfWeek')['fraud'].count()
    fraud_transactions.plot(kind='bar', figsize=(10,5), title='Fraud Transactions by Day of Week')
    plt.show()
    #check if transaction fraud is higher any specific hour of the day
    fraud_transactions = transactions[transactions['fraud'] == 1].groupby('HourOfDay')['fraud'].count()
    fraud_transactions.plot(kind='bar', figsize=(10,5), title='Fraud Transactions by Hour of Day')
    plt.show()
    #check is transaction is higher for any specific merchant
    # Filter only the fraudulent transactions and then group by merchantId and count
    fraud_transactions = transactions[transactions['fraud'] == 1].groupby('merchantId').size()
    # Plot the top 10 merchants with the highest count of fraudulent transactions
    fraud_transactions.sort_values(ascending=False).head(10).plot(kind='barh', figsize=(10,5), title='Number of fraudulent transactions by merchant ID')
    plt.show()
    # Filter only the fraudulent transactions and then group by accountNumber and count
    fraud_transactions = transactions[transactions['fraud'] == 1].groupby('accountNumber').size()
    # Plot the top 10 accounts with the highest count of fraudulent transactions
    fraud_transactions.sort_values(ascending=False).head(10).plot(kind='bar', figsize=(10,5), title='Number of fraudulent transactions by Account Number')
    plt.show()
    # Filter only the fraudulent transactions and then group by posEntryMode and count
    fraud_transactions = transactions[transactions['fraud'] == 1].groupby('posEntryMode').size()
    # Select the top 10 POS Entry Modes with the highest count of fraudulent transactions
    top_fraud_posEntryModes = fraud_transactions.nlargest(10)
    # Create a pie chart
    fig, ax = plt.subplots()
    ax.pie(top_fraud_posEntryModes, labels = top_fraud_posEntryModes.index, autopct='%1.1f%%')
    plt.title('Proportion of fraudulent transactions by POS Entry Mode')
    plt.show()

def label_encoding(transactions):
    #Need to change the label encoding for the categorical variables to numerical variables for accountNumber, MerchantID, DayOfWeek
    le = LabelEncoder()
    transactions['accountNumber'] = le.fit_transform(transactions['accountNumber'])
    transactions['merchantId'] = le.fit_transform(transactions['merchantId'])
    transactions['DayOfWeek'] = le.fit_transform(transactions['DayOfWeek'])
    return transactions

def split_data_to_feature_target(transactions):
    #split the data into features and target
    X = transactions.drop('fraud', axis=1).values
    y = transactions['fraud'].values
    return X, y

def scale_data(X):
    #scaling the data to bring all the features to the same level of magnitude
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X

def split_data(X, y):
    #split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def smote(X_train, y_train):
    #SMOTE (Synthetic Minority Oversampling Technique) is an oversampling technique that generates synthetic samples from the minority class.
    #SMOTE works by selecting examples that are close in the feature space, drawing a line between the examples in the feature space and drawing a new sample at a point along that line.
    #SMOTE is used to obtain a synthetically class-balanced or nearly class-balanced training set, which is then used to train the classifier.
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    return X_train, y_train

def train_model(X_train, y_train):
    # Define the parameter grid
    best_params = {
        'n_estimators': 100,
        'max_features': 'sqrt',
        'max_depth': None,
        'class_weight': None,
        'criterion': 'entropy'
    }

    # Initialize a Random Forest classifier
    rf_best = RandomForestClassifier(**best_params, random_state=42)

    rf_best.fit(X_train, y_train)
    #add tqdm to show progress bar

    return rf_best

def predict_model(rf_best, X_test):
    # Predict the labels of the test data: y_pred
    y_pred = rf_best.predict(X_test)
    return y_pred, rf_best

def evaluate_model(y_test, y_pred):
    # Generate the confusion matrix, classification report, and roc_auc_score
    print('ROC_AUC_SCORE: ',roc_auc_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    

def plot_confusion_matrix(y_test, y_pred):
    # Generate the confusion matrix, classification report, and roc_auc_score
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt=".0f", linewidths=0.5, square = True, cmap = 'Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix', size = 15)
    plt.show()

def plot_roc_curve(y_test, y_pred, rf_best, X_test):
    #Getting the prediction probabilities using the random forest classifier
    y_pred_prob = rf_best.predict_proba(X_test)[:,1]

    # Getting the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

    #Getting the area under the ROC curve
    roc_auc = auc(fpr, tpr)
    print('ROC AUC: %0.2f' % roc_auc)

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


def main():
    # Load the data
    data = load_data('transactions_obf.csv', 'labels_obf.csv')
    
    # Get features and target
    transactions = get_features_and_target(data[0], data[1])
    
    #Preprocess the data
    transactions = preprocess(transactions)
    
    #Plot fraud by hour
    plot_fraud(transactions)

    #Label encoding
    label_encoding(transactions)

    #Split data into features and target
    X, y = split_data_to_feature_target(transactions)

    #Scale the data
    X = scale_data(X)

    #Split the data into train and test
    X_train, X_test, y_train, y_test = split_data(X, y)

    #SMOTE
    X_train, y_train = smote(X_train, y_train)

    #Train the model
    best_params = train_model(X_train, y_train)

    #Fit the model
    y_pred, rf_best = predict_model(best_params, X_test)

    #Evaluate the model
    evaluate_model(y_test, y_pred)

    #Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)

    #Plot ROC curve
    plot_roc_curve(y_test, y_pred, rf_best, X_test)

# This line is the entry point to the script
if __name__ == '__main__':
    main()
