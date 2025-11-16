# 🎓 Student Final Score Predictor

A Streamlit-based machine learning web application that predicts a student’s **Final Score** using academic, personal, and behavioral factors.  
The app trains a **Linear Regression model**, selects the most significant features, and provides **real-time predictions** through an intuitive and interactive interface.

---

## 🚀 Features

✔️ Real-time prediction of student final score  
✔️ Automatic feature selection using **SelectKBest**  
✔️ **Linear Regression** model for accurate predictions  
✔️ Automatic encoding of categorical features  
✔️ Modern Streamlit UI with sliders & dropdowns  
✔️ Clean, responsive, and user-friendly design  

---

## 📂 Project Structure

📁 Student-Final-Score-Predictor
│

   ├── app.py # Streamlit web application
   
   ├── Task_students_performance_dataset.xlsx
   
   ├── student_performance_model.pkl # Saved Linear Regression model
   
   ├── README.md # Project documentation
   
   └── requirements.txt # Dependencies (optional)


## 🧠 How It Works

📥 Loads the dataset  
🔠 Encodes categorical variables using LabelEncoder  
🎯 Selects top features using SelectKBest (f_regression)  
🔀 Splits data into training & test sets  
🧪 Trains a Linear Regression model  
💾 Saves the trained model using joblib  
🖥️ Displays an interactive UI for user inputs  
🎓 Predicts the final score instantly  

---

## 🛠️ Technologies Used

- Python  
- Streamlit  
- Pandas  
- Scikit-learn  
- Joblib  

---

## ▶️ How to Run the App

### 1️⃣ Clone the repository

git clone https://github.com/kaviyadharshini2805/student-final-score-predictor.git
cd student-final-score-predictor
### 2️⃣ Install dependencies

Copy code
pip install -r requirements.txt
### 3️⃣ Run the Streamlit App

Copy code
streamlit run app.py


### 🎯 Usage
Adjust the sliders or dropdowns for student details

Click Predict Final Score

Instantly view the predicted score in the result card
