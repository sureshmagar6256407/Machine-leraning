import pandas as pd 
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