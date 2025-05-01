# NeuroNexus: Titanic Survival Prediction

## Overview
This project predicts whether passengers survived the Titanic disaster using machine learning. Built for the NeuroNexus internship.

## Dataset
- Source: [Kaggle](https://www.kaggle.com/datasets/brendan45774/test-file)
- Columns: PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked

## Streamlit Web App
- **URL**: https://neuronexus-titanic-survival-prediction.streamlit.app/
- Interactive app to input passenger details, predict survival using the Random Forest model, view EDA plots, and display sample predictions.
- **Run Locally**: `streamlit run app.py`

## Steps
1. **EDA**: Visualized survival patterns by gender, class, age, and fare.
2. **Preprocessing**: Handled missing values, encoded categorical features, and scaled numerical features.
3. **Feature Engineering**: Extracted titles, created `family_size` and `has_cabin` features.
4. **Modeling**: Trained Logistic Regression and Random Forest models.
5. **Evaluation**: Achieved ~80% accuracy with Random Forest.

## Results
- Random Forest: ~80% accuracy, ~0.75 F1-score for `Survived=1`.
- Key predictors: Sex, Pclass, Title.

## How to Run
1. Clone the repo: `git clone https://github.com/yourusername/NeuroNexus`
2. Install dependencies: `pip install -r requirements.txt`
3. Open `notebooks/titanic_survival_prediction.ipynb` in Jupyter.
