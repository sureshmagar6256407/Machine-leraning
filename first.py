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
"""


"""
import pandas as pd 
from sklearn.model_selection import train_test_split  
from sklearn.linear_model  import LinearRegression  
from sklearn.metrics import mean_squared_error , r2_score  
import joblib as jlb  

data = {
"Student": [1,2,3,4,5],
"Hours_Studied": [5,8,3,10,6],
"Attendance_%": [80,90,70,95,85],
"Previous_Score": [70,80,60,85,75],
"Final_Score": [75,85,65,90,80]
}

df  = pd.DataFrame(data)  
# print(df)

#features   and target  
X  = df[["Hours_Studied","Attendance_%","Previous_Score"]]  
y = df["Final_Score"]

#split the data 
X_train,X_test, y_train,y_test  = train_test_split(X,y , test_size=0.2, random_state= 42)   

#mode train  
model  = LinearRegression()
model.fit(X_train,y_train)
  

#model predict  
y1 = model.predict(X_test) 
# print(y1)
# print(y_test)

y2 = model.predict([[6,90,85]])
print(y2)
# print(model.coef_)
# print(model.intercept_)

mse  = mean_squared_error(y_test,y1) 
print(mse)

""" 


email  = { 
    101 : {"Email": None, "password" : None}
}

try : 
    Email = input("Enter your email :: ")  
    passwrod  = input("Enter your password")
    if Email not in email.values() : 
        if Email.endswith("@gmial.com") and passwrod : 
            if Email.count("@") == 1 and Email.count('.') == 1 : 
                email["Email"] = email 
                email["password"] = passwrod
            else : 
                print("@ and . is only one check")
        else : 
            print("email must be endswith @gmail.com")
    else : 
        print("the id is alredy in the email ")
    

except Exception as e  : 
    print(e)