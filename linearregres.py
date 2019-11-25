import pandas as pd 
import quandl,math
import numpy as np 
from sklearn import preprocessing,svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df= quandl.get('WIKI/GOOGL')
pd.set_option('display.max_columns', None)
df['hl_pct']=(df['Adj. High']-df['Adj. Close'])/df['Adj. Close']*100
df['pct_change']=(df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100
df=df[['hl_pct','Adj. Close','Adj. Open','pct_change']]
forecast_col='Adj. Close'
df.fillna(-9999,inplace=True)
forecast_out=int(math.ceil(0.01*len(df)))
print(forecast_out)
df['label']=df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
x=np.array(df.drop(['label'],1))
y=np.array(df['label'])
x=preprocessing.scale(x)
df.dropna(inplace=True)
y=np.array(df['label'])
print(len(x),len(y))
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
clf= LinearRegression()
clf.fit(x_train,y_train)
accuracy=clf.score(x_test,y_test)
print(accuracy)







