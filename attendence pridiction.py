import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
data = pd.read_csv("attendance.csv")
X = data[['day_of_week', 'weather', 'previous_attendance', 'test_near']]
y = data['attendance_drop']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(" Attendance Drop Prediction System ")
print(f"Model Accuracy : {accuracy*100:.2f}%")
print("-----------------------------------")
new_data = [[2, 1, 68, 1]]  # Tue, Rain, 68%, Test near
probability = model.predict_proba(new_data)[0][1] * 100
if probability < 40:
    risk = "LOW"
elif probability < 70:
    risk = "MEDIUM"
else:
    risk = "HIGH"
print("Prediction Details:")
print(f"Attendance Drop Probability : {probability:.2f}%")
print(f"Risk Level                 : {risk}")
if probability >= 50:
    print("Final Decision      : Attendance Drop Expected")
else:
    print("Final Decision      : Attendance Likely Stable")

print("*************************************************************")
