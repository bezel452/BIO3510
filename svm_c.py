from sklearn import svm
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import eli5
from eli5.sklearn import PermutationImportance
import pickle
import matplotlib.pyplot as plt
from matplotlib import rc

file_path = "Data/clean_data_final.csv"

data = pd.read_csv(file_path)
def work(epoch, flag):
    train_data = data.sample(frac=0.8,random_state=epoch,axis=0)
    test_data = data[~data.index.isin(train_data.index)]

    train_y = train_data["overall_survival"]
    train_x = train_data.drop(columns=["overall_survival"])

    test_y = test_data["overall_survival"]
    test_x = test_data.drop(columns=["overall_survival"])

    model = svm.SVC(probability=True)

    model.fit(train_x, train_y)

    y_pred = model.predict(test_x)

    acc = sum(y_pred == test_y) / len(y_pred)

    print("准确率:", acc)
    if flag == True:
        return acc
    pre_y = model.predict_proba(test_x)
    pre_y = pre_y[:,1]
    fpr, tpr, thresholds = roc_curve(test_y, pre_y, pos_label=1)
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
    plt.savefig("ROC_SVM.png")
    plt.show()

    return acc
'''
imp = PermutationImportance(model).fit(test_x, test_y)
eli5.show_weights(imp, feature_names = test_x.columns.tolist())
'''

if __name__ == '__main__':
    work(983, False)

