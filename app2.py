import pandas as pd
import streamlit as st
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load and preprocess data
@st.cache_data
def load_data():
    a = pd.read_csv("diamonds.csv")
    
    # Remove duplicates
    a.drop_duplicates(inplace=True)
    
    # Encode categorical variables
    ob = a.select_dtypes(include="object")
    la = LabelEncoder()
    
    for i in ob.keys():
        a[i] = la.fit_transform(ob[i])
    
    # Save encoder
    pickle.dump(la, open("encode.pkl", "wb"))
    
    # Drop unnecessary columns
    a.drop(columns=["table", "x", "y", "z"], inplace=True)
    a.rename(columns={"cut": "quality"}, inplace=True)
    
    return a

# Train and save model
@st.cache_resource
def train_model(data):
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, train_size=0.8)  # Fixed train_size
    
    rfr = RandomForestRegressor()
    rfr.fit(x_train, y_train)
    
    # Save model
    pickle.dump(rfr, open("model.pkl", "wb"))
    
    return rfr

# Main app
def main():
    st.title("💎 Diamond Price Prediction")
    st.sidebar.header("Made By Subhadip 😎")
    
    # Load data and train model (or load if already trained)
    try:
        data = load_data()
        model = train_model(data)
        encode = pickle.load(open("encode.pkl", "rb"))
    except:
        st.error("Error loading data or model. Please ensure diamonds.csv is available.")
        return
    
    # Inputs
    Carat = st.slider("Carat", 0.0, 10.0, 1.0)
    depth = st.slider("Depth", 0.0, 80.0, 60.0)
    
    # Quality mapping
    quality_dict = {"Fair": 0, "Good": 1, "Ideal": 2, "Premium": 3, "Very Good": 4}
    quality = st.selectbox("Choose Quality", list(quality_dict.keys()))
    quality_val = quality_dict[quality]
    
    # Color mapping (based on typical diamond color grades)
    color_list = {"Dodger Blue": 0, "Emerald": 1, "Fuchsia": 2, "Gray": 3, "Honeydew": 4, "Ice Blue": 5, "Jade": 6}
    color = st.selectbox("Choose Color", list(color_list.keys()))
    color_encoded = color_list[color]
    
    # Clarity information
    st.sidebar.write('''
    | Clarity  | Simple Meaning             | Notes                                              |
    | -------- | -------------------------- | -------------------------------------------------- |
    | **SI1**  | Small inclusions           | Tiny marks, usually not seen without magnification |
    | **SI2**  | Small inclusions           | Slightly bigger marks, might be seen with the eye  |
    | **VS1**  | Very small inclusions      | Hard to see even under magnification               |
    | **VS2**  | Very small inclusions      | Small marks, mostly not visible to the eye         |
    | **VVS2** | Very very small inclusions | Almost impossible to see                           |
    | **IF**   | Internally flawless        | No internal marks, only tiny surface marks         |
    | **I1**   | Included                   | Marks visible to the eye, may affect shine         |
    ''')
    
    # Clarity selection
    clarity_list = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "IF"]
    clarity = st.selectbox("Choose Clarity", clarity_list)
    clarity_encoded = encode.transform([clarity])[0]
    
    # Prediction
    if st.button("Predict Price"):
        try:
            prediction = model.predict([[Carat, depth, quality_val, color_encoded, clarity_encoded]])
            st.success(f"Predicted Diamond Price: ${prediction[0]:.2f}")
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    main()