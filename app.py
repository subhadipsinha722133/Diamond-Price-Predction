import streamlit as st
import pickle

st.title("Diamond Price Prediction")

# Load model and encoder
model = pickle.load(open("model.pkl", "rb"))
encode = pickle.load(open("encode.pkl", "rb"))  # Assuming it's a LabelEncoder or OneHotEncoder

# Inputs
Carat = st.slider("Carat", 0.0, 10.0, 1.0)
depth = st.slider("Depth", 0, 80, 60)

st.sidebar.write('''| Clarity  | Simple Meaning             | Notes                                              |
| -------- | -------------------------- | -------------------------------------------------- |
| **SI1**  | Small inclusions           | Tiny marks, usually not seen without magnification |
| **SI2**  | Small inclusions           | Slightly bigger marks, might be seen with the eye  |
| **VS1**  | Very small inclusions      | Hard to see even under magnification               |
| **VS2**  | Very small inclusions      | Small marks, mostly not visible to the eye         |
| **VVS2** | Very very small inclusions | Almost impossible to see                           |
| **IF**   | Internally flawless        | No internal marks, only tiny surface marks         |
| **I1**   | Included                   | Marks visible to the eye, may affect shine         |
''')




# Quality mapping
quality_dict = {"Ideal": 2, "Premium": 3, "Very Good": 4, "Good": 1, "Fair": 0}
quality = st.selectbox("Choose Quality", list(quality_dict.keys()))
quality_val = quality_dict[quality]



# # Color and Clarity
# color_list = ["Gray", "Emerald", "Fuchsia", "Honeydew", "Dodger Blue", "Ice Blue", "Jade"]
# color = st.selectbox("Choose Color", color_list)
# color_encoded = encode.transform([color])[0]  # Encode string to numeric

# Color and Clarity
color_list = {"Gray":3, "Emerald":1, "Fuchsia":2, "Honeydew":4, "Dodger Blue":0, "Ice Blue":5, "Jade":6}
color = st.selectbox("Choose Color", list(color_list.keys()))
# color_encoded = encode.transform([color])[0]  # Encode string to numeric
color_encoded=color_list[color]


clarity_list = ["SI1","SI2","VS1","VS2","VVS2","IF","I1"]
clarity = st.selectbox("Choose Clarity", clarity_list)
clarity_encoded = encode.transform([clarity])[0]

# Prediction
if st.button("Predict"):
    prediction = model.predict([[Carat, depth, quality_val, color_encoded, clarity_encoded]])
    st.success(f"Predicted Diamond Price: ${prediction[0]:.2f}")

