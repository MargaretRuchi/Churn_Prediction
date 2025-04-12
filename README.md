# Churn_Prediction
A Streamlit-based web app for predicting telecom customer churn using a trained XGBoost model.

---

## Features

- Predict whether a customer is likely to churn or stay.
- User-friendly interface with input fields for customer data.
- Built with Streamlit for fast deployment and visualization.
- Uses SMOTE for data balancing.
- Model trained with XGBoost and fine-tuned using GridSearchCV.

---

## Demo

To run the app locally:

```bash
# Clone the repository
git clone https://github.com/your-username/Churn_Prediction.git
cd Churn_Prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
