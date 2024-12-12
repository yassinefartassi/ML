import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Sample dataset
# Features: [weight(grams), length(cm), sweetness(1-10), has_seeds(0/1)]
fruits_data = [
    [150, 7, 8, 1],  # Apple
    [120, 6, 7, 1],  # Pear
    [80, 12, 9, 1],  # Banana
    [200, 8, 8, 1],  # Orange
    [300, 15, 7, 1], # Mango
]

vegetables_data = [
    [100, 15, 2, 0],  # Carrot
    [200, 20, 3, 0],  # Cucumber
    [300, 10, 1, 0],  # Potato
    [150, 12, 2, 0],  # Broccoli
    [250, 25, 1, 0],  # Zucchini
]

# Create labels (0 for vegetables, 1 for fruits)
labels = [1] * len(fruits_data) + [0] * len(vegetables_data)

# Combine all data
X = np.array(fruits_data + vegetables_data)
y = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Function to predict new items
def predict_fruit_or_vegetable(weight, length, sweetness, has_seeds):
    features = np.array([[weight, length, sweetness, has_seeds]])
    prediction = model.predict(features)
    return "Fruit" if prediction[0] == 1 else "Vegetable"

# Test the model accuracy
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Example predictions
test_samples = [
    [140, 7, 8, 1],  # Should predict fruit (similar to apple)
    [120, 18, 2, 0]  # Should predict vegetable (similar to carrot)
]

print("\nExample predictions:")
for sample in test_samples:
    result = predict_fruit_or_vegetable(*sample)
    print(f"Weight: {sample[0]}g, Length: {sample[1]}cm, Sweetness: {sample[2]}/10, Has Seeds: {sample[3]}")
    print(f"Prediction: {result}\n")
