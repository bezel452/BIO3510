from sklearn import ensemble
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.inspection import permutation_importance
import csv

file_path = "Data/clean_data_final.csv"

data = pd.read_csv(file_path)

train_data = data.sample(frac=0.8,random_state=0,axis=0)
test_data = data[~data.index.isin(train_data.index)]

train_y = train_data["overall_survival"]
train_x = train_data.drop(columns=["overall_survival"])

test_y = test_data["overall_survival"]
test_x = test_data.drop(columns=["overall_survival"])

model = ensemble.RandomForestClassifier(n_estimators=400)

model.fit(train_x, train_y)

y_pred = model.predict(test_x)

acc = sum(y_pred == test_y) / len(y_pred)

print("准确率：",acc)

# calculate the AUC and draw the ROC figure

fpr, tpr, thresholds = roc_curve(test_y, y_pred, pos_label=1)
a = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='red',
         lw=2, label='ROC curve (area = %0.2f)' % a)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0, 1.05])
plt.ylim([0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("AUC-ROC figure")
plt.legend(loc="lower right")
plt.savefig("ROC_RF.png")
plt.show()

# feature importance
res = permutation_importance(model, test_x, test_y)
imp = res.importances_mean
p = list(zip(test_x.columns.tolist(), imp))
p = sorted(p, key=lambda x: x[1], reverse=True)
print(p)
with open("featimpor_RF.csv", "w") as f:
    writer = csv.writer(f)
    header = ["Features", "Weights"]
    writer.writerow(header)
    writer.writerows(p)