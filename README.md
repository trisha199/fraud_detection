![](https://resources.cdn.seon.io/uploads/2021/12/Credit_Card_Fraud_Detection.svg)

![](https://img.shields.io/badge/python-3.8-blue)
![sickit](https://img.shields.io/badge/scikit--learn-compatible-orange)


## Introduction to Credit Card Fraud Detection Project
In the era of digital transactions, credit card fraud represents a significant challenge for financial institutions and consumers alike. The Credit Card Fraud Detection Project aims to tackle this pervasive issue by employing advanced data analysis and machine learning algorithms. By analyzing transactional data, the project seeks to identify and predict fraudulent activities, thereby mitigating risks and protecting financial assets.

At the heart of the project is a predictive model that has been trained using the Synthetic Minority Over-sampling Technique (SMOTE) to address class imbalance, a common problem in fraud detection where fraudulent transactions are much rarer than legitimate ones. The model's performance is evaluated using robust metrics such as the Receiver Operating Characteristic (ROC) curve and the Area Under the Curve (AUC), ensuring that both the prevalence and the detection of fraud are accurately measured.

With a focus on precision and recall, the project strives to fine-tune the balance between correctly identifying fraudulent transactions (true positives) and not misclassifying legitimate transactions as fraud (true negatives). The ultimate goal is to create a reliable system that can be integrated into transaction processing workflows to provide real-time fraud detection, thereby enhancing security for all stakeholders in the credit card ecosystem.

The `fraud_detection.py` script is designed for detecting fraudulent activities in financial transactions. It uses Python to analyze transaction data, applying machine learning techniques to identify potential fraud.

## Features
- **Data Handling**: Utilizes `pandas` and `numpy` for efficient data manipulation and analysis.
- **Visualization Tools**: Employs `matplotlib` and `seaborn` for insightful data visualization, aiding in identifying unusual transaction patterns.
- **Machine Learning**: Implements `KNeighborsClassifier` from `sklearn` for fraud detection, with supporting features for data scaling and model evaluation.
- **Performance Evaluation**: Leverages ROC curve and AUC metrics from `sklearn` for assessing the effectiveness of the fraud detection model.

## Requirements
- Python 3.x
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`

## Installation
Install the required Python libraries using the following command:
```bash
pip install pandas numpy matplotlib seaborn sklearn
```
## Visualizations Overview
The fraud_detection.py script generates several key visualizations that are critical for understanding the nature of the data and the performance of the fraud detection model:

### Confusion Matrix: This visualization provides insight into the accuracy of the fraud detection model. It contrasts the true positives, true negatives, false positives, and false negatives, offering a clear picture of the model's classification performance.
![](https://github.com/abh2050/fraud_detection/blob/main/confusion_matrix.png)

### ROC Curve: The Receiver Operating Characteristic curve illustrates the true positive rate against the false positive rate at various threshold settings. The area under the curve (AUC) is a measure of the model's ability to distinguish between fraudulent and non-fraudulent transactions.
![](https://github.com/abh2050/fraud_detection/blob/main/roc_curve.png)

### Fraud by Merchant ID: This bar chart ranks merchants by the number of fraudulent transactions associated with their ID, highlighting potential sources of fraud.
![](https://github.com/abh2050/fraud_detection/blob/main/fraud_by_merchant.png)

### Number of Fraudulent Transactions by Account Number: Similar to the merchant ID visualization, this bar chart identifies accounts with a higher number of fraudulent transactions, which can be useful for pinpointing high-risk accounts.
![](https://github.com/abh2050/fraud_detection/blob/main/fraudbyaccount.png)

### Fraud Transactions by Day of Week: This chart displays the total number of fraudulent transactions for each day of the week, offering insights into any patterns or trends in the timing of fraudulent activities.
![](https://github.com/abh2050/fraud_detection/blob/main/fraudbydayofweek.png)

These visualizations are an essential component of the fraud detection system, aiding in the interpretation of the model's findings and in the strategic planning of fraud prevention measures.

## Usage Guide
1. **Data Preparation**: Prepare your dataset in two CSV files - `transactions_obf.csv` for transaction data and `labels.csv` for fraud labels.
2. **Running the Script**: Execute the script in your Python environment. It processes the data, trains the model, and evaluates its performance.
3. **Output**: The script outputs a fraud detection model, which can be used to classify new transactions.

## Customization
- The script can be modified for different datasets or to integrate other machine learning models.
- Adjust visualization and model parameters for tailored analysis and improved accuracy.

## Conclusion
The model demonstrates high overall accuracy, as indicated by the near-perfect classification of non-fraudulent transactions (class 0) with a precision and recall of approximately 1.00. This suggests the model is very effective at identifying legitimate transactions.

For the fraudulent transactions (class 1), the precision is high at 0.83, indicating that when the model predicts fraud, it is correct 83% of the time. However, the recall is lower at 0.57, meaning that the model only identifies 57% of all fraudulent transactions. The F1-score of 0.68 for fraudulent transactions indicates a moderate balance between precision and recall, but there is room for improvement, especially in terms of recall.

The ROC AUC score obtained from the test set is 0.7829, which is a good score and indicates that the model has a strong ability to distinguish between the two classes. However, the AUC value of 0.9656 seems to be conflicting with the ROC AUC score. If the 0.9656 AUC pertains to the training phase or another model evaluation, it suggests an excellent discriminatory ability there.

In conclusion, while the model excels at identifying legitimate transactions, efforts could be made to improve the detection of fraudulent ones, particularly in increasing the recall without significantly reducing precision. The disparity between the AUC scores needs clarification, but overall, the model shows promise for credit card fraud detection and could benefit from further tuning and validation.

## Disclaimer
- This script is a starting point for fraud detection analysis. Tailor it to your specific data and requirements for production use.
- The accuracy of fraud detection depends on the quality and characteristics of the input data.
