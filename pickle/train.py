import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle

print("imported everything")

df=pd.read_csv('diabetes.csv')
print(df.head())
X=df.drop(['Outcome'],axis=1)
y=df['Outcome']

X_train, X_test, y_train, y_test=model_selection.train_test_split(X,y,train_size=0.75,random_state=101)

#fit the model
log_reg=LogisticRegression()
log_reg.fit(X_train,y_train)

# Accuracy
result=log_reg.score(X_test,y_test)
print(f'the accuracy of the model is {result}')

#save the model...have to import joblib
pickle.dump(log_reg,open('diabetes_80.pkl','wb'))