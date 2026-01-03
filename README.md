# ATTENDENCE-PREDICTION
This project predicts attendance drop using machine learning. It analyzes past attendance, weather conditions, day of the week, and exam proximity to classify whether attendance is likely to fall. The system also provides a probability-based risk level to support academic planning.
#Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- VS Code
# Dataset Features
- day_of_week  
- weather  
- previous_attendance  
- test_near  
- attendance_drop (target)
# How It Works
1. Loads attendance dataset from a CSV file  
2. Splits data into training and testing sets  
3. Trains a Logistic Regression model  
4. Predicts attendance drop probability  
5. Displays accuracy and risk level  
