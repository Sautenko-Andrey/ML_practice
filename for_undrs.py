import pandas as pd
import os

data={'chaus':[197,85,35],'sautenko':[190,78,33]}
data=pd.DataFrame(data=data,index=['heeight','weight','age'])
print(data)
print('---------------------------------------')

players=['chaus','tereh','makak','dinsh']
data=[[45,'LEV'],[89,'Gigant'],[75,'DGMA'],[83,'Donbass']]
data=pd.DataFrame(data=data,index=players,columns=['rank','team'])
print(data)
print('----------')
print(data.loc['makak'])