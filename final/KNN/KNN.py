from sklearn.model_selection  import cross_val_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc 

# 读取数据
data = pd.read_csv("../clean_data_final.csv")
X = data.iloc[:, 1:].values
Y = data.iloc[:, 0].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=2)
k_accuracy = []
k_range = range(1, 100)
#循环，取k=1到k=31，查看误差效果
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,Y_train)
    Y_predict = knn.predict(X_test)
#     print(Y_predict)
    acu=sum(Y_predict==Y_test)/len(Y_test)
    k_accuracy.append(acu)
    print(acu)

#画图，x轴为k值，y值为误差值
plt.figure(figsize=(4,4))
plt.plot(k_range, k_accuracy)
plt.xlabel('Value of K for KNN')
plt.ylabel('accuracy')
plt.show()

k = 32
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=2)
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, Y_train)
Y_predict = knn.predict(X_test)
acu=sum(Y_test==Y_predict)/len(Y_test)
print(acu)
# y_pred_proba = knn.predict_proba(X_test)[:, 1]
y_pred_proba = knn.predict_proba(X_test)[:, 1]
fpr,tpr,threshold = roc_curve(Y_test, y_pred_proba)
roc_auc = auc(fpr,tpr)

plt.figure()
lw = 2
# plt.figure(figsize=(5,5))
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.plot(fpr, tpr,
#          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
# plt.plot([0, 1], [0, 1], lw=lw, linestyle='--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUC-ROC figure')
plt.legend(loc="lower right")
plt.show()

from sklearn.inspection import permutation_importance
# importance
result = permutation_importance(knn, X_test, Y_test, n_repeats=1, random_state=2)
importance = result.importances_mean
# 取features
features = list(data.columns)
features = features[1:]
# 结果
feature_importances = list(zip(features, importance))
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

print(feature_importances)
