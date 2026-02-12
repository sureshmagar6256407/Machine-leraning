"""
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error , r2_score

data = { 
    "day":[1,2,3,4,5,6,7,8,9],
    "price":[4500,5000,6000,6500,7000,75000,4400,2000,4900]
}
df = pd.DataFrame(data)
#features 
X = df[["day"]]
y   = df["price"]

#train and testing 
X_train, X_test , y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42 )

#model train 
model = LinearRegression()
model.fit(X_train,y_train)

#model predict   
y_predict  = model.predict(X_test)
print(y_predict)
print(y_test)

y1 = model.predict([[100]])
print(y1)
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

"""
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error , r2_score  
data = {
    "Size_sqft": [500, 800, 1000, 1200, 1500, 1800, 2000, 2200, 2500, 3000],
    "Price": [5000000, 8000000, 10000000, 12000000, 15000000, 
              18000000, 20000000, 22000000, 25000000, 30000000]
}

df = pd.DataFrame(data)

#features and target  
X = df[["Size_sqft"]]
y= df["Price"]

#split data  
X_train , X_test , y_train, y_test = train_test_split (X,y , test_size= 0.2 ,random_state= 42)  

#train model  
model  = LinearRegression() 
model.fit(X_train , y_train)   

#predict   
# pre = model.predict(X_test)
# print(pre)
# print(y_test)
pre = model.predict([[2700]])
print(pre)
print(model.coef_)
print(model.intercept_)
print('\n')
pre2= model.predict(X_test)  
print(pre2)
# r2 score
y_pred  = model.predict(X_test)  
print("R2 scores:" , r2_score(y_test,y_pred))

#12 day predict 

#mse   calculate   
mse = mean_squared_error(y_test, y_pred)
print(f"the mse is {mse}")
"""


"""
import pandas as pd  
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics  import mean_squared_error , r2_score  

data = {
    "Size_sqft": [500, 800, 1000, 1200, 1500, 1800, 2000, 2200, 2500, 3000],
    "Price": [5000000, 8000000, 10000000, 12000000, 15000000, 
              18000000, 20000000, 22000000, 25000000, 30000000]
}

df = pd.DataFrame(data)

#features and target 
X  = df[["Size_sqft"]]
y = df["Price"]

#split data 
X_train , X_test , y_train , y_test  = train_test_split(X,y,test_size= 0.2 ,random_state= 42)  

#model train  
model  =LinearRegression() 
model.fit(X_train , y_train)

#model predict   
y1 = model.predict(X_test)
print(y1, '\n')
print(y_test)

#re score 
print(f"r2 score of model is : {r2_score(y_test,y1)}")

# y2 = model.predict([[3100]])
# print(y2)

#mse  
mse  = mean_squared_error(y_test , y1)
print(f"the mse is {mse}")
"""

# import pandas as pd 
# from sklearn.model_selection import train_test_split  
# from sklearn.linear_model import LinearRegression  
# from sklearn.metrics import  mean_squared_error ,r2_score 


"""
data  = { 
    "day" :[1,2,3,4,5,6,7] ,
    "goldPrice" : [200000,204000,205000,215000,225000,230000,240000]
}
df = pd.DataFrame(data) 

#features and target 
X  = df[["day"]]
y = df["goldPrice"]  


#split data  
X_train, X_test , y_train , y_test = train_test_split (X,y, test_size= 0.2, random_state= 42)  


#mode train   
model = LinearRegression()
model.fit(X_train ,y_train) 

#mode predict   
y1 = model.predict([[8]])
print(y1 ,'\n')

ytrain = model.predict(X_test)  
print(ytrain)
print(y_test)

#r2 score 
print(f"the model r2 score is {r2_score(y_test ,  ytrain)}")
""" 


"""
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model  import LinearRegression  
from sklearn.metrics  import mean_squared_error , r2_score 

data  =  { 
    "day": [1,2,3,4,5,6]  , 
    "price" :[500,1000,1200,1900,2500,3700]
}
df = pd.DataFrame(data)
#features and target  
X = df[["day"]]
y   =df["price"]  

#split data 
X_train , X_test , y_train, y_test  = train_test_split(X , y , test_size= 0.2 ,random_state= 42)  

#train model 
model  = LinearRegression()
model.fit(X_train ,y_train )

#predict  
y1  = model.predict(X_test) 
print(y1)  
print(y_test)

print(f"r2 score is {r2_score(y_test , y1)}")
"""