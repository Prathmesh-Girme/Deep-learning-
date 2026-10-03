#_________________________________________________________________
Border = "_"*60
#
# Deep Learning Pipeline
# 
# 1. Read the data from csv
# 2 .Data Analysis
# 3  Pre-processing
# 4. Train Test Split
# 5. Feature Scaling
# 6. FNN Model training
# 7. Model Evaluation
# 8. Graphical Representation
# 9. model Preserve
# 10.Model Loading and preserve
# 11. Test unseen Data
#___________________________________________________________________


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


#Step 1 : - Import data and read
print(Border)
print(" 1. Read the data from csv")
print(Border)

data = pd.read_csv("placement_data.csv")

print("Complete Dataset : ")
print(data)




#step 2 : - EDA 
print(Border)
print("2. Data Analysis (EDA)")
print(Border)

print("COLUMN NAMES : ")
print(data.columns)

print("Shape of dataset : ")
print(data.shape)

print("Statistical Summary")
print(data.describe())



#step 3 - preprocessing
print(Border)
print("3. Preprocessing")
print(Border)


X = data[['Aptitude', 'Coding', 'Communication', 'Academics', 'Internship']]
Y = data['Placed']

print("Input Features : ")
print(X.head())



print("Target :")
print(Y.head())



#step 4 : Train test split
print(Border)
print("4. Train Test Split")
print(Border)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

print("Training input Shape : ", X_train.shape)
print("Testing  input Shape : ", X_test.shape)
print("Training output shape : ", Y_train.shape)
print("Testing output shape  : ", Y_test.shape)




#step 5 = Feature scaling
print(Border)
print("5 . Feature Scaling")
print(Border)

scalar = StandardScaler()

X_train_scaled = scalar.fit_transform(X_train)

X_test_scaled = scalar.transform(X_test)

print("Scaled training data : ")
print(X_train_scaled[:5])




#step 6 : FNN Model Creation
print(Border)
print("6 : FNN Model Training")
print(Border)

Model = MLPClassifier(
    hidden_layer_sizes=(8,4),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42

)

print(Model)




print(Border)
print("Train the model ...")
print(Border)

Model.fit(X_train_scaled, Y_train)

print("Model Training Completed")




#step 7 : Model Evaluation
print(Border)
print("7. Model Evaluation")
print(Border)

Y_pred = Model.predict( X_test_scaled) 

accuracy = accuracy_score(Y_test, Y_pred)


print("Accuracy of the model : ", accuracy * 100)


cm = confusion_matrix(Y_test, Y_pred)

print("Confusion Matrix : ")
print(cm)

print("predict the probability of the model : ")
Y_prob = Model.predict_proba(X_test_scaled)
# print("Probability : ", Y_prob)

print("First 5 probabilities : ", Y_prob[:5])

#step 8 : Graphical Representation
print(Border)
print("8. Graphical Representation")
print(Border)

# ---- Graph 1 : Confusion Matrix Heatmap ----
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Placed", "Placed"])
disp.plot(ax=ax, cmap="Blues", colorbar=True)
plt.title(f"Confusion Matrix (Accuracy: {accuracy*100:.1f}%)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# ---- Graph 2 : Training Loss Curve ----
# MLPClassifier automatically records loss per iteration in loss_curve_
plt.figure(figsize=(8, 5))
plt.plot(Model.loss_curve_, color="red")
plt.title("Training Loss Curve (MLPClassifier)")
plt.xlabel("Iteration (Epoch)")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150)
plt.show()

# ---- Graph 3 : Prediction Probability Distribution ----
plt.figure(figsize=(8, 5))
plt.hist(Y_prob[:, 1], bins=15, color="skyblue", edgecolor="black")
plt.axvline(0.5, color="red", linestyle="--", label="Decision threshold (0.5)")
plt.title("Predicted Probability of 'Placed' Across Test Samples")
plt.xlabel("Predicted Probability of Placement")
plt.ylabel("Number of Students")
plt.legend()
plt.tight_layout()
plt.savefig("probability_distribution.png", dpi=150)
plt.show()

# ---- Graph 4 : Feature Scaling Before vs After ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].boxplot(
    [X_train['Aptitude'], X_train['Coding'], X_train['Communication'],
     X_train['Academics'], X_train['Internship']],
    tick_labels=['Aptitude', 'Coding', 'Comm.', 'Academics', 'Internship']
)
axes[0].set_title("Before Scaling (Raw Features)")
axes[0].set_ylabel("Value")
axes[0].tick_params(axis='x', rotation=30)

axes[1].boxplot(
    [X_train_scaled[:, 0], X_train_scaled[:, 1], X_train_scaled[:, 2],
     X_train_scaled[:, 3], X_train_scaled[:, 4]],
    tick_labels=['Aptitude', 'Coding', 'Comm.', 'Academics', 'Internship']
)
axes[1].set_title("After StandardScaler")
axes[1].set_ylabel("Scaled Value")
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig("scaling_comparison.png", dpi=150)
plt.show()

print("All graphs generated and saved successfully.")


#step 9 : Model Preserve
print(Border)
print("9. Model Preserve")
print(Border)

joblib.dump(Model, 'placement_fnn_model.pkl')
joblib.dump(scalar, 'placement_fnn_scaler.pkl')

print("Model and Scaler are preserved/dumpedS successfully")



#python pikel file == pkl file


#step 10 : Model Loading and preserve
print(Border)
print("10. Model Loading and preserve")
print(Border)

loaded_model = joblib.load('placement_fnn_model.pkl')
loaded_scalar = joblib.load('placement_fnn_scaler.pkl')

print("Model and Scaler are loaded successfully")


#step 11 : Test unseen Data
# Aptitude : 70
# coding :  75
# Communication :  80
# Academics :   85
# Internship : 1

print(Border)
print("11. Test unseen Data")
print(Border)


new_student = pd.DataFrame([[70,  75,  80,   85, 1]] , columns=['Aptitude', 'Coding', 'Communication', 'Academics', 'Internship'])


new_student_scaled = loaded_scalar.transform(new_student)

new_student_pred = loaded_model.predict(new_student_scaled)
print("Prediction for new student : ", new_student_pred)

new_prob = loaded_model.predict_proba(new_student_scaled)
print("Probability for new student : ", new_prob)

print("New students data : ")
print(new_student)

print("prediction probability : ", new_prob)


if new_student_pred[0] == 1:
    print("The student is likely to be placed")
else:
    print("The student is unlikely to be placed")