import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
# load data
data = pd.read_csv('creditcard.csv')

# separate legitimate and fraudulent transactions
valid = data[data.Class == 0]
fraud = data[data.Class == 1]

# undersample legitimate transactions to balance the classes
valid_sample = valid.sample(n=len(fraud), random_state=2)
data = pd.concat([valid_sample, fraud], axis=0)

# split data into training and testing sets
X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Create a dictionary to store model names and their corresponding models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier()
}

# Create a dictionary to store accuracy scores
accuracy_scores = {}

# Train and evaluate models
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores[model_name] = accuracy

color_map = {
    "Logistic Regression": "green",
    "Random Forest": "blue",
    "Decision Tree": "red"
}

# Streamlit app
st.title("Credit Card Fraud Detection Different Algorithms")
st.subheader("Model Comparison")

# Display accuracy scores
st.write("Model Accuracy Scores:")
st.write(accuracy_scores)


# train logistic regression model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# evaluate model performance
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

st.title("Credit Card Fraud Detection Using RandomForestClassifier Model")
st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")

# create input fields for user to enter feature values
input_df = st.text_input('Input All features')
# Assuming your input is tab-separated
input_df_lst = input_df.split('\t')

# Convert strings to floats
try:
    features = [float(val) for val in input_df_lst]
except ValueError:
    st.write("Please enter valid numerical values separated by tabs.")

# create a button to submit input and get prediction
submit = st.button("Submit")

if submit:
    # get input feature values
    features = np.array(input_df_lst, dtype=np.float64)
    # make prediction
    prediction = model.predict(features.reshape(1,-1))
    # display result
    if prediction[0] == 0:
        st.title("Legitimate transaction")
    else:
        st.title("Fraudulent transaction")


# Plot bar chart for accuracy comparison
fig, ax = plt.subplots(figsize=(5, 2))
bars = ax.bar(accuracy_scores.keys(), accuracy_scores.values(), color=[color_map[model] for model in accuracy_scores.keys()])
ax.set_ylabel('Accuracy')
ax.set_title('Model Accuracy Comparison')
plt.xticks()
st.pyplot(fig)

st.write("Classification Report:")
for model_name, model in models.items():
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.write(f"Classification Report for {model_name}:")
    st.dataframe(report_df.style.background_gradient(cmap='coolwarm'))
