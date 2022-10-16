import pickle
# load the model
model=pickle.load(open("diabetes_80.pkl",'rb'))
prediction=model.predict([[1,93,70,31,0,3,1,23]])[0]
print(prediction)
print("donettt")