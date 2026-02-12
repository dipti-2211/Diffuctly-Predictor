import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# --- 1. THE DATA & MODEL ---
# This is the same logic from your predictor.py
data = {
    'Topic_Code': [1, 2, 1, 3, 2, 3, 1, 2], 
    'Lines_of_Code': [12, 45, 15, 80, 50, 120, 25, 35],
    'Difficulty': [0, 2, 0, 2, 1, 2, 1, 1] 
}
df = pd.DataFrame(data)

X = df[['Topic_Code', 'Lines_of_Code']].values
y = df['Difficulty'].values

model = DecisionTreeClassifier()
model.fit(X, y)

diff_map = {0: "Easy", 1: "Medium", 2: "Hard"}
topic_map = {"Array": 1, "Tree": 2, "DP": 3}

# --- 2. STREAMLIT UI ---
st.set_page_config(page_title="LeetCode AI", page_icon="💻")

st.title("🚀 LeetCode Difficulty Predictor")
st.write("A 'Vibe Coder' project to predict problem difficulty using Machine Learning.")

# User Inputs
st.divider()
col1, col2 = st.columns(2)

with col1:
    selected_topic = st.selectbox("Select the Topic:", list(topic_map.keys()))

with col2:
    lines = st.number_input("Lines of Code:", min_value=1, max_value=500, value=30)

# Prediction Logic
if st.button("Predict Difficulty"):
    # Convert text topic back to the numerical code (1, 2, or 3)
    topic_code = topic_map[selected_topic]
    
    # Run the model
    user_input = np.array([[topic_code, lines]])
    prediction = model.predict(user_input)
    result = diff_map[prediction[0]]
    
    # Display Result with different colors
    if result == "Easy":
        st.success(f"Result: {result} 😊")
    elif result == "Medium":
        st.warning(f"Result: {result} 😐")
    else:
        st.error(f"Result: {result} 🤯")

st.divider()
