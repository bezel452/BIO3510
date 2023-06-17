# %%
from sklearn.ensemble import RandomForestRegressor 
import pandas as pd
import numpy as np

# 加载数据
data = pd.read_csv('/home/yangfu/courses/Biobigdata/proj/BIO3510/FCNN/clean_data2.csv')

# %%
list(data.columns)

# %%
clinical = data.iloc[:,2:28]
target = data["death_from_cancer"]

# %%
rf = RandomForestRegressor(n_estimators=200, random_state=0)

# %%
rf.fit(clinical, target)
importances = rf.feature_importances_

# %%
# 将特征重要性与列名一起打印出来，并选择具有最高重要性的前N个列
n = 5
important_columns = []
for col, score in zip(clinical.columns, importances):
    print(f'{col}: {score}')
    if len(important_columns) < n:
        important_columns.append(col)

# %%
clinical.columns[importances.argsort()]

# %%
clinical.columns[np.where(importances>0.035)]

# %%
rf.score(clinical, target)


