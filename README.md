# codebasis-project
import pandas as pd
import numpy as np
df=pd.read_excel("/content/reg cod.xlsx")
df.dtypes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
model=LinearRegression()
# Prepare the features (2022, 2023, 2024) and target ('EV 2030')
X = df[['State_code','CAGR','Year-2022', 'Year-2023', 'Year-2024']]  # Features: EV sales for 2022, 2023, 2024
y = df['EV 2030']  # Target: EV sales in 2030
model.fit(X,y)
model.predict(X)
#Check model coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
p=mean_squared_error(df['EV 2030'],model.predict(X))
print(p)
f=r2_score(df['EV 2030'],model.predict(X))
print(f)
from sklearn.ensemble import RandomForestRegressor
ke=RandomForestRegressor()
ke.fit(X,y)
ke.predict(X)
sa=mean_squared_error(df['EV 2030'],ke.predict(X))
print(sa)
sd=r2_score(df['EV 2030'],ke.predict(X))
print(sd)
import matplotlib.pyplot as plt 
plt.plot(df['EV 2030'],model.predict(X))
plt.scatter(df['EV 2030'],model.predict(X))
plt.show()
def Gradient_Decent(X,y):
  n=10
  learning_rate=0.01
  m_curr=b_curr=0
  iterations=1000
  for i in range(iterations):
    y_predicted=m_curr*X+b_curr
    cost=1/n*sum([val**2 for val in (y-y_predicted)])
    md=-(2/n)*sum(X*(y-y_predicted))
    bd=-(2/n)*sum(y-y_predicted)
    m_curr=m_curr-learning_rate*md
    b_curr=b_curr-learning_rate*bd  
    print("m {},b {},cost {},iteration {}".format(m_curr,b_curr,cost,i))
    import pickle
    with open('model_pickle','wb') as f:
  pickle.dump(model,f)
  with open('model_pickle','rb') as f:
  dp=pickle.load(f)
  with open('model_pickle1','wb') as f:
  pickle.dump(ke,f)
  with open('model_pickle1','rb') as f:
  dp=pickle.load(f)
  import pandas as pd
from sklearn.preprocessing import OneHotEncoder
le=OneHotEncoder(sparse_output=False)
data=pd.DataFrame(df)
encode=le.fit_transform(data[['STATE']])
# Convert the encoded states into a DataFrame
encoded_df = pd.DataFrame(encode, columns=le.get_feature_names_out(['STATE']))

# Concatenate the original DataFrame with the encoded DataFrame
df = pd.concat([df, encoded_df], axis=1)

# Display the DataFrame with one-hot encoded state columns
print(df)
