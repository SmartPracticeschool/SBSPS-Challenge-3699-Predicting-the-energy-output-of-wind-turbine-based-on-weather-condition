import pandas as pd
import numpy as np
import xlsxwriter
np=np.ones(3)

df=pd.DataFrame(np)
print(df)
data=pd.ExcelWriter('frompy.xlsx',engine='xlsxwriter')
df.to_excel(data,sheet_name='Sheet1')
data.save()