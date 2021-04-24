import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.model_selection import train_test_split
#%%

from sklearn.externals import joblib

#%%
df=pd.read_csv('train.csv')

df.head(5)

y=df['price_range']
x=df.drop(['price_range'],axis=1)


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


#feature selection
gbc=GradientBoostingClassifier()
rfc=RandomForestClassifier()

rfe_selector = RFE(estimator=gbc, n_features_to_select=11, step=10, verbose=5)
rfe_selector.fit(x, y)
rfe_support = rfe_selector.get_support()
rfe_feature = x.loc[:,rfe_support].columns.tolist()
print(rfe_feature)

#['battery_power', 'fc', 'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w'


rfe_selector = RFE(estimator=rfc, n_features_to_select=11, step=10, verbose=5)
rfe_selector.fit(x, y)
rfe_support = rfe_selector.get_support()
rfe_feature_2 = x.loc[:,rfe_support].columns.tolist()
print(rfe_feature_2)
#['battery_power', 'clock_speed', 'int_memory', 'mobile_wt', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time']




x=df[rfe_feature]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
from sklearn.metrics import accuracy_score
gbc.fit(x_train,y_train)
preds1=gbc.predict(x_test)
score1=accuracy_score(y_test,preds1)
print(score1)
#90%

x=df[rfe_feature_2]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
from sklearn.metrics import accuracy_score
rfc.fit(x_train,y_train)
preds2=gbc.predict(x_test)
score2=accuracy_score(y_test,preds2)

#%%
model = open("mobile_prcie_prediction.pkl","wb")
joblib.dump(gbc,model)
model.close()

#75%

#%%

#so we will we using gradient boosting classifier for our 

print(df[rfe_feature].columns)