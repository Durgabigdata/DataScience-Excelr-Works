import streamlit as st
import pickle


with open("C:\\Users\\Admin\\Downloads\\vectorizer.pkl", 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

with open("C:\\Users\\Admin\\Downloads\\rf_model.pkl", 'rb') as model_file:
    rf_model = pickle.load(model_file)


# Streamlit setup
st.title("Drug Classification NLP Model")
st.write("Predicts the Drug Category based on Patient reviews")


review = st.text_area("Enter the patient review:",placeholder="Enter your Review here...")


if st.button("Predict"):
    if review.strip():
       
        transformed_review = vectorizer.transform([review])
        
        
        prediction = rf_model.predict(transformed_review)
        
       
        st.write(f"The predicted drug category is: **{prediction[0]}**")
        
       
        
    else:
        st.warning("Please enter a valid review.")  


st.sidebar.write("### Model Information")
st.sidebar.write("This model was trained using Random Forest and is designed to classify patient reviews into categories: Depression, Diabetes, and High Blood Pressure.")


st.sidebar.write("OUTPUT")

st.sidebar.write("[100] means Depression")
st.sidebar.write("[010] means Diabetes")
st.sidebar.write("[001] means High Blood Pressure")

st.sidebar.write("Model Accuracy: 86%")

# Separator for better UI layout

st.markdown("---")

# Instructions for users

st.markdown('<div class="instructions"><h4>How It Works:</h4><ol><li>Enter a patient\'s review describing their condition.</li><li>Click the \'Predict\' button to predict the condition.</li><li>Get recommended drugs based on the classified condition.</li></ol></div>', unsafe_allow_html=True)

