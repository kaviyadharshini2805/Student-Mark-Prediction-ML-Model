import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="🎓 Student Final Score Predictor", layout="wide")

st.title("🎓 Live Predictor - Student Final Score")

data = pd.read_excel("Task_students_performance_dataset.xlsx")

X = data.drop('Final_Score', axis=1)
y = data['Final_Score']

for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

selector = SelectKBest(score_func=f_regression, k=min(8, X.shape[1]))
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
X = X[selected_features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
joblib.dump(model, "student_performance_model.pkl")

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📊 Enter Student Details")

cols = st.columns(2)
inputs = {}

for i, col_name in enumerate(X.columns):
    with cols[i % 2]:
        if X[col_name].nunique() < 10:
            inputs[col_name] = st.selectbox(col_name, options=X[col_name].unique())
        else:
            inputs[col_name] = st.slider(
                col_name,
                float(X[col_name].min()),
                float(X[col_name].max()),
                float(X[col_name].mean())
            )

input_df = pd.DataFrame([inputs])

if st.button("🔮 Predict Final Score"):
    for col in input_df.columns:
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
    input_df = input_df.fillna(0)
    prediction = model.predict(input_df)[0]
    st.success(f"🎯 Predicted Final Score: {prediction:.2f}")
