🔍 Rock vs Mine Classification
This project is a Machine Learning-based classification model that predicts whether a given sonar signal is reflected from a rock or a mine. It uses logistic regression and random forest algorithms trained on the Sonar dataset to achieve high accuracy. A Streamlit app is also included for an easy-to-use interactive interface.

📁 Project Structure
├── logistic_regression_model.pkl 
├── random_forest_model.pkl 
├── rock_mine.ipynb 
├── sonar_mine_prediction_app.py 
├── requirements.txt 
└── README.md

⚙️ Setup Instructions
✅ 1. Clone the Repository
git clone https://github.com/adjha/Rock-vs-Mine-Classification.git
cd Rock-vs-Mine-Classification

For Windows:
python -m venv venv
venv\Scripts\activate

For Mac/Linux:
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt


📌 How to Run:
▶️ Jupyter Notebook (Model Training & Evaluation)
Run this to explore data, train models, and test them:

#jupyter notebook rock_mine.ipynb
This notebook:
*Loads and preprocesses the dataset.
*Trains two models: Logistic Regression and Random Forest.
*Saves trained models as .pkl files.

▶️ Streamlit Web App
To launch the web interface for prediction:
streamlit run sonar_mine_prediction_app.py

The app:
*Loads the trained .pkl model
*Accepts user input for 60 sonar attributes
*Predicts whether the signal is from a Rock or a Mine

🏆 Achievements
*End-to-end ML pipeline with training, evaluation, model serialization, and UI.
*Streamlit-based web interface for non-technical usage.
*Clean and modular code design for ease of extension.
