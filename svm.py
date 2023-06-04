from sklearn import svm

label = []
feature = []
with open("Data/breast cancer.csv", "r") as f:
    line = f.readline()
    line = f.readline()
    while line:
        t = line.split(',')
        label.append(float(t[24]))
        feature.append([float(i) for i in t[31:520]])
        line = f.readline()


train_x = feature[:1600]
train_y = label[:1600]
test_x = feature[1601:]
test_y = label[1601:]

model = svm.SVC()

model.fit(train_x, train_y)

y_pred = model.predict(test_x)

acc = sum(y_pred == test_y) / len(y_pred)

print(acc)

