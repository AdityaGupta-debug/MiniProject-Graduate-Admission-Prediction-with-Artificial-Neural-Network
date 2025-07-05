import streamlit as st
import pickle
import tensorflow as tf
import pandas as pd 

model = tf.keras.models.load_model('model.h5')

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.title("University Admission Predictor")
st.header("Predict Your Chance of Admission to University")
GRE_Score = st.number_input('GRE Score', min_value=0, max_value=340, value=300)
TOEFL_Score = st.number_input('TOEFL Score', min_value=0, max_value=120, value=100)
University_Rating = st.number_input('University Rating', min_value=1, max_value=5, value=3)
SOP = st.number_input('Statement of Purpose (SOP)', min_value=1.0, max_value=5.0, value=3.0)
LOR = st.number_input('Letter of Recommendation (LOR)', min_value=1.0, max_value=5.0, value=3.0)
CGPA = st.number_input('CGPA', min_value=0.0, max_value=10.0, value=8.0)
Research = st.selectbox('Research Experience (0 = No, 1 = Yes)', options=[0, 1], index=0)

if st.button('Predict Admission Chance'):
    input_data = pd.DataFrame({
        'GRE Score': [GRE_Score],
        'TOEFL Score': [TOEFL_Score],
        'University Rating': [University_Rating],
        'SOP': [SOP],
        'LOR': [LOR],
        'CGPA': [CGPA],
        'Research': [Research]
    })

    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)

    st.success(f'🎓 Predicted Chance of Admission: {prediction[0][0] * 100:.2f}%')