# --- STEP 0: Imports ---
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# --- STEP 1: Pandas (The Data) ---
# We define our small dataset
data = {
    'Topic_Code': [1, 2, 1, 3, 2, 3], # 1: Arrays, 2: Trees, 3: DP
    'Lines_of_Code': [15, 45, 12, 80, 50, 120],
    'Difficulty': [0, 1, 0, 1, 1, 1]  # 0: Easy, 1: Hard
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
test_problem = np.array([[2, 10]])
prediction = model.predict(test_problem)

# Output the result
result = "Hard" if prediction[0] == 1 else "Easy"
print(f"Project Result: The model thinks a 10-line Tree problem is {result}.")