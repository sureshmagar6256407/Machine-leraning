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

"""
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error , r2_score
data  = { 
    "day" : [1,2,3,4,5,6,7,8,9], 
    "Price" : [1500,2000,3100,3900,2000,4200,50000,7000,8000]
}
df  =pd.DataFrame (data)  
# split the data   
X = df[["day"]] 
y  = df["Price"]  
#train    
X_train,X_test, y_train, y_test = train_test_split(X,y, train_size=0.2,random_state=42)


# modeltrain
model  = LinearRegression()  
model.fit(X_train, y_train )  

#model predict  
y_predict = model.predict(X_test) 
# print(y_predict)
# print(y_test)
# # print(y_test)
# print(model.coef_)
# print(model.intercept_)
y1 = model.predict([[20]])
print(y1)
"""

"""
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import  LinearRegression 
from sklearn.metrics import mean_squared_error , r2_score

data = { 
    "Date" :[1,2,3,4,5,6,7,8,9,10] , 
    "GoldRatePerTola" :[300000,286600,290300,304700,295200,291000,300500,304600,305500 ,306600]
}

df= pd.DataFrame(data)

#split the data  
X = df[["Date"]]
y = df["GoldRatePerTola"]

#train the data 
X_train,X_test, y_train, y_test  = train_test_split(X,y, test_size=0.2,train_size=0.8 ,random_state=42)

#model train 
model  = LinearRegression()
model.fit(X_train,y_train)

#predict  
# pre  = model.predict(X_test)
# print(pre)
# print(y_test)

day11 = model.predict([[11]])
print(day11)
print(model.coef_)
print(model.intercept_)
"""