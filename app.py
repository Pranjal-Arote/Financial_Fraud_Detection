# app.py
from flask import Flask, render_template, request, jsonify
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

app = Flask(__name__)

# Load dataset
df = pd.read_csv("C:/DataSet/online_fraud_detection.csv")

# Remove rows with any missing values
df = df.dropna()

# Map 'isFraud' values to 'No Fraud' and 'Fraud'
df['isFraud'] = df['isFraud'].map({0: 'No Fraud', 1: 'Fraud'})

# Map 'type' values to numerical values
df['type'] = df['type'].map({'PAYMENT': 1, 'TRANSFER': 4, 'CASH_OUT': 2, 'DEBIT': 5, 'CASH_IN': 3})

# Apply one-hot encoding to the 'type' column
encoder = OneHotEncoder()
type_encoded = encoder.fit_transform(df[['type']])

# Convert the encoded sparse matrix to a DataFrame
type_encoded_df = pd.DataFrame.sparse.from_spmatrix(type_encoded)

# Concatenate the encoded columns with the DataFrame
df = pd.concat([df, type_encoded_df], axis=1)

# Drop the original 'type' column
df.drop('type', axis=1, inplace=True)

# Split data into features and target
x = df[['amount', 'oldbalanceOrg', 'newbalanceOrig', 0, 1, 2, 3, 4]]  # Assuming the encoded columns are 0, 1, 2, 3, and 4
y = df['isFraud']

# Convert column names to strings
x.columns = x.columns.astype(str)

# Train-test split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20, random_state=42)

# Train model
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        amount = float(request.form['amount'])
        oldbalanceOrg = float(request.form['oldbalanceOrg'])
        newbalanceOrig = float(request.form['newbalanceOrig'])

        # Make prediction
        prediction = model.predict([[amount, oldbalanceOrg, newbalanceOrig, 0, 0, 0, 0, 0]])

        return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
