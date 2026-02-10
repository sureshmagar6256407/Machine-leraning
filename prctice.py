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