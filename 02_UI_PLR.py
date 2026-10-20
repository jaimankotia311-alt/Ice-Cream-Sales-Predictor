import joblib
import streamlit as st

model = joblib.load(r"C:\Users\Jay Mankotia\Desktop\AI ML\04 Ice Cream PLR\plr.joblib")
poly = joblib.load(r"C:\Users\Jay Mankotia\Desktop\AI ML\04 Ice Cream PLR\poly.joblib")

st.title("ICE CREAM SALE PREDICTOR")
st.markdown("# 🍦 Ice Cream Sales Predictor")
st.markdown("### Predict ice cream sales based on temperature!")
st.markdown("Enter the temperature below and click **PREDICT** to estimate sales.")
temp  = st.text_input("TEMPERATURE", placeholder="Write your temperature : ")

if st.button("PREDICT"):
    new_temp = poly.fit_transform([[float(temp)]])
    prediction = model.predict(new_temp)
    st.write (f"The predicted sale, would be around:  {round(prediction[0])}")




#  py -m streamlit run "C:\Users\Jay Mankotia\Desktop\AI ML\04 Ice Cream PLR\02_UI_PLR.py"    


