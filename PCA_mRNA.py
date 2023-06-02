import pandas as pd
from sklearn.decomposition import PCA

# 返回一个元组，第一项是对mRNA信息进行PCA后的data（删除原来列，PCA列添加在后面）
# 第二项是PCA各个主成分中对应mRNA的占比

def PCA_mRNA(num_components, data = pd.read_csv("breast cancer.csv")):

    data = pd.read_csv("breast cancer.csv")
    mRNAs = data.loc[:,"brca1":"ugt2b7"]
    print(mRNAs.head())

    num_miss = mRNAs.isnull().sum()
    print(f"Number of missing values: {sum(num_miss)}")

    pca = PCA(n_components=num_components)
    principle_components = pca.fit_transform(mRNAs)
    print(f"Shape of principle components: {principle_components.shape}")

    original_features = pca.components_
    print(f"Principle components -- Original features:\n{original_features}")

    data_pca = data.drop(columns=data.loc[:,"brca1":"ugt2b7"].columns)
    for i in range(num_components):
        data_pca[f"Principle_Component_{i+1}"] = principle_components[:,i]

    return data_pca, original_features