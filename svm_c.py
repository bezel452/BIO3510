from sklearn import svm
import pandas as pd
import eli5
from eli5.sklearn import PermutationImportance

file_path = "Data/clean_data_final.csv"

data = pd.read_csv(file_path)
def work(epoch):
    train_data = data.sample(frac=0.8,random_state=epoch,axis=0)
    test_data = data[~data.index.isin(train_data.index)]

    train_y = train_data["overall_survival"]
    train_x = train_data.drop(columns=["overall_survival"])

    test_y = test_data["overall_survival"]
    test_x = test_data.drop(columns=["overall_survival"])

    model = svm.SVC()

    model.fit(train_x, train_y)

    y_pred = model.predict(test_x)

    acc = sum(y_pred == test_y) / len(y_pred)

    print("准确率:", acc)
    return acc
'''
imp = PermutationImportance(model).fit(test_x, test_y)
eli5.show_weights(imp, feature_names = test_x.columns.tolist())
'''

if __name__ == '__main__':
    work(0)