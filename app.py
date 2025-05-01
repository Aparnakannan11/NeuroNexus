import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Titanic Survival Prediction", layout="wide")

try:
    model = joblib.load('rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Model or scaler file not found. Please run the notebook to generate rf_model.pkl and scaler.pkl.")
    st.stop()

try:
    df = pd.read_csv('data/tested.csv')
except FileNotFoundError:
    st.error("data/tested.csv not found. Ensure the file is in the correct directory.")
    st.stop()

# Streamlit app
st.title("Titanic Survival Prediction")

st.sidebar.header("Enter Passenger Details")
pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3], index=2)
sex = st.sidebar.selectbox("Sex", ["male", "female"], index=0)
age = st.sidebar.slider("Age", 0, 100, 30)
sibsp = st.sidebar.slider("Siblings/Spouses Aboard", 0, 8, 0)
parch = st.sidebar.slider("Parents/Children Aboard", 0, 6, 0)
fare = st.sidebar.slider("Fare", 0.0, 500.0, 50.0)
embarked = st.sidebar.selectbox("Embarked", ["C", "Q", "S"], index=2)
has_cabin = st.sidebar.checkbox("Has Cabin", value=False)

sex_encoded = 1 if sex == "female" else 0
embarked_q = 1 if embarked == "Q" else 0
embarked_s = 1 if embarked == "S" else 0
family_size = sibsp + parch + 1

title = st.sidebar.selectbox("Title", ["Mr", "Miss", "Mrs", "Rare"], index=0)

title_miss = 1 if title == "Miss" else 0
title_mr = 1 if title == "Mr" else 0
title_mrs = 1 if title == "Mrs" else 0
title_rare = 1 if title == "Rare" else 0

input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex_encoded],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'has_cabin': [int(has_cabin)],
    'Embarked_Q': [embarked_q],
    'Embarked_S': [embarked_s],
    'Title_Miss': [title_miss],
    'Title_Mr': [title_mr],
    'Title_Mrs': [title_mrs],
    'Title_Rare': [title_rare],
    'family_size': [family_size]
})

try:
    input_data[['Age', 'Fare', 'family_size']] = scaler.transform(input_data[['Age', 'Fare', 'family_size']])
except Exception as e:
    st.error(f"Error scaling input data: {e}")
    st.stop()

try:
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]
except Exception as e:
    st.error(f"Error making prediction: {e}")
    st.stop()

st.header("Prediction")
st.write(f"Survival Prediction: **{'Survived' if prediction == 1 else 'Did Not Survive'}**")
st.write(f"Probability of Survival: **{prob:.2%}**")

st.subheader("Input Features")
st.write(input_data)

st.header("Exploratory Data Analysis")
st.subheader("Survival by Gender")
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x='Survived', hue='Sex', data=df, ax=ax)
ax.set_title('Survival by Gender')
st.pyplot(fig)

st.subheader("Survival by Passenger Class")
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x='Survived', hue='Pclass', data=df, ax=ax)
ax.set_title('Survival by Passenger Class')
st.pyplot(fig)

st.header("Sample Predictions")
try:
    submission = pd.read_csv('submission.csv')
    st.write(submission.head())
except FileNotFoundError:
    st.error("submission.csv not found. Ensure the notebook has generated it.")
