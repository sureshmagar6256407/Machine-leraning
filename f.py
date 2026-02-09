import pandas as pd 
data  = { 
    "Name" : ["suresh","Tekam","rahul" ],
    "age" : [30,50,20]
}
df = pd.DataFrame(data)

minage = df.groupby("Name")["age"].min()
print(f"minimal age is {minage}")

largeage  =df.groupby("Name")["age"].max()
print(f"max age is {largeage}")