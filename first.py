"""import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error ,r2_score


data = {
"Size_sqft": [1000,1200,1500,1800,2000],
"Price_k": [200,220,270,310,340]
}
df = pd.DataFrame(data)
# print(df)

#features and target
X = df[["Size_sqft"]]#features
y = df["Price_k"] #target

#split Data :
X_train, X_test ,y_train,y_test =train_test_split(X,y , test_size=0.2 ,random_state=42)

#train Model
model  = LinearRegression()
model.fit(X_train,y_train)

#preditct 
y_pred = model.predict(X_test)
print(y_pred)
print(y_test)

y1 =model.predict([[5000]])
print(y1)

print(model.coef_)#slope
print(model.intercept_)#intercept
"""



import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import joblib as jlb

data = {

    "Size_sqft": [1000, 1200, 1500, 1800, 2000],
    "Bedrooms":  [2,    2,    3,    3,    4],
    "Price_k":   [200, 220,  270,  310,  340]

}

df = pd.DataFrame(data)
# print(df)

#features and target 
X = df[["Size_sqft", "Bedrooms"]]
y= df["Price_k"]

#split   
X_train , X_test , y_train, y_test =train_test_split(X,y,test_size=0.2, random_state=42)
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

#model train 
model = LinearRegression()
model.fit(X_train,y_train)

#model prediction  
y_pred  =  model.predict(X_test)
print(y_pred)
print(y_test)
# print(y)
# print(y_test)
# print(model.coef_)
# print(model.intercept_)

#r2 score  
mse  = mean_squared_error(y_test,y_pred)
print(f"the mse is {mse}")
print(f"the r2 score is {r2_score( y_test,y_pred)}")

# prediction own values  
y1 = model.predict([[5000,10]])
print(y1)

#save model  
jlb.dump(model,"Firstml.pkl") 
