
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Features:
# Study_Hours, Attendance, Assignment_Score

X = [
    [1, 40, 30],
    [2, 50, 35],
    [3, 60, 40],
    [4, 65, 55],
    [5, 70, 55],
    [6, 75, 65],
    [7, 80, 70],
    [2, 45, 25],
    [8, 90, 85],
    [1, 35, 20],
    [3, 55, 20],
    [4, 65, 20],
    [5, 70, 55],
    [6, 75, 60],
    [7, 85, 75]
]

# 0 = Fail
# 1 = Pass

Y = [0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1]


# Split data into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.3,
    random_state=42
)


# Create Neural Network
model = MLPClassifier(
    hidden_layer_sizes=(5,),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)


# Train the model
model.fit(X_train, Y_train)


# Make predictions
Y_Pred = model.predict(X_test)


# Display results
print("Actual output    :", Y_test)
print("Predicted output :", Y_Pred)


# Calculate accuracy
accuracy = accuracy_score(Y_test, Y_Pred)

print("Accuracy of deep learning model is:", accuracy)




