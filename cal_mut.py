import pandas as pd

data = pd.read_csv("Data/breast cancer.csv")

index = data.columns.tolist()[520:]

maxn = 0

dic = {}

for idx in index:
    t = data[idx]
    cnt = sum(t != '0')
    dic[idx] = cnt
    if maxn < cnt:
        maxn = cnt
        name = idx

dic = sorted(dic.items(), key=lambda x:x[1],reverse=True)

print("Maximum: %d; Gene: %s"%(maxn, name))
print(dic[:20])
