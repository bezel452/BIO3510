from sklearn import ensemble
import pandas as pd

file_path = "Data/clean_data_fi.csv"

data = pd.read_csv(file_path)

train_data = data.sample(frac=0.8,random_state=0,axis=0)
test_data = data[~data.index.isin(train_data.index)]

train_y = train_data["overall_survival"]
train_x = train_data.drop(columns=["overall_survival"])

test_y = test_data["overall_survival"]
test_x = test_data.drop(columns=["overall_survival"])

model = ensemble.RandomForestClassifier()

model.fit(train_x, train_y)

y_pred = model.predict(test_x)

acc = sum(y_pred == test_y) / len(y_pred)

print("准确率：",acc)

# print(model.feature_importances_)