import joblib
# load the model
model=joblib.load("diabetes_801.pkl")
prediction=model.predict([[1,93,70,31,0,30.4,0.315,23]])[0]
print(prediction)
print("donettt")