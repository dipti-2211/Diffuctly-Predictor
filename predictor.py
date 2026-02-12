# --- STEP 0: Imports ---
import pandas as pd 
import numpy as np 
from sklearn.tree import DecisionTreeClassifier 

# --- STEP 1: Pandas (The Data) ---
# We define our small dataset
data = {
    'Topic_Code': [1, 2, 1, 3, 2, 3, 1, 2], 
    'Lines_of_Code': [12, 45, 15, 80, 50, 120, 25, 35],
    # 0: Easy, 1: Medium, 2: Hard
    'Difficulty': [0, 2, 0, 2, 1, 2, 1, 1] 
}

df = pd.DataFrame(data)

# --- STEP 2: NumPy (The Matrix) ---
# Splitting data into X (input features) and y (output target)
X = df[['Topic_Code', 'Lines_of_Code']].values
y = df['Difficulty'].values

# --- STEP 3: Scikit-Learn (The Model) ---
# Initialize the "Brain"
model = DecisionTreeClassifier()

# Fit (Train) the tree - this is where it learns the patterns
model.fit(X, y)

# --- STEP 4: Testing/Inference ---
# Let's test a new problem: A Tree problem (2) with only 10 lines of code
test_problem = np.array([[1, 28]])
prediction = model.predict(test_problem)

diff_map = {0:"Easy" , 1:"Medium" , 2 : "Hard"} 

print("\n--- LeetCode Difficulty Predictor ---")
try:
    # This takes input from your terminal
    topic = int(input("Enter Topic Code (1:Array, 2:Tree, 3:DP): "))
    lines = int(input("Enter Lines of Code: "))

    # Prepare the input for the model
    user_test = np.array([[topic, lines]])
    
    # Predict!
    prediction = model.predict(user_test)
    result = diff_map[prediction[0]]

    print(f"\n🚀 Analysis Complete!")
    print(f"The AI predicts this is a {result} problem.")

except ValueError:
    print("❌ Error: Please enter numbers only (e.g., 1 for Topic and 30 for Lines).")