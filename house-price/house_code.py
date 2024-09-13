import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

# 读取数据
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 对SalePrice进行对数变换
train['SalePrice'] = np.log1p(train['SalePrice'])

# 删除 ID 列
train.drop(["Id"], axis=1, inplace=True)
Id_test_list = test["Id"].tolist()
test.drop(["Id"], axis=1, inplace=True)

# 提取数值特征
df_train_num = train.select_dtypes(exclude=["object"])
df_test_num = test.select_dtypes(include=[np.number])

# 设置筛选门槛
sel = VarianceThreshold(threshold=0.05)

# 对训练集进行拟合和转换（排除目标变量列）
df_train_num_reduced = sel.fit_transform(df_train_num.drop("SalePrice", axis=1))
df_train_num = pd.DataFrame(df_train_num_reduced, columns=df_train_num.drop("SalePrice", axis=1).columns)

# 对测试集进行转换
df_test_num_reduced = sel.transform(df_test_num)
df_test_num = pd.DataFrame(df_test_num_reduced, columns=df_train_num.columns)

# 强相关性的特征 (corr > 0.7)
df_train_corr = train.corr()["SalePrice"][:-1]
strong_features = df_train_corr[abs(df_train_corr) >= 0.7].index.tolist()
strong_features.append("SalePrice")
df_train_strong = df_train_num[strong_features]
df_test_strong = df_test_num[strong_features[:-1]]

# 中等相关特征 (0.5 < corr < 0.7)
moderate_features = df_train_corr[(abs(df_train_corr) < 0.7) & (abs(df_train_corr) >= 0.5)].index.tolist()
moderate_features.append("SalePrice")
df_train_moderate = df_train_num[moderate_features]
df_test_moderate = df_test_num[moderate_features[:-1]]

# 弱相关特征 (0.3 < corr < 0.5)
weak_features = df_train_corr[(abs(df_train_corr) < 0.5) & (abs(df_train_corr) >= 0.3)].index.tolist()
weak_features.append("SalePrice")
df_train_weak = df_train_num[weak_features]
df_test_weak = df_test_num[weak_features[:-1]]

# 合并所有特征
all_features = strong_features[:-1] + moderate_features[:-1] + weak_features[:-1]
df_train_final = df_train_num[all_features + ["SalePrice"]]
df_test_final = df_test_num[all_features]

# 对缺失值进行插补
my_imputer = SimpleImputer(strategy="median")
df_train_imputed = pd.DataFrame(my_imputer.fit_transform(df_train_final.drop("SalePrice", axis=1)), columns=df_train_final.drop("SalePrice", axis=1).columns)
df_test_imputed = pd.DataFrame(my_imputer.transform(df_test_final), columns=df_test_final.columns)

# 添加目标变量
df_train_imputed["SalePrice"] = df_train_final["SalePrice"].values

# 删除特征中有异常值的列（基于你的需求）
df_train_imputed.drop(["LotFrontage"], axis=1, inplace=True)
df_test_imputed.drop(["LotFrontage"], axis=1, inplace=True)

# 处理分类特征
categorical_features = [i for i in train.columns if train.dtypes[i] == "object"]
categorical_features.append("SalePrice")

df_train_categ = train[categorical_features]
df_test_categ = test[categorical_features[:-1]]

# 删除那些“单一”的分类变量
cols_to_drop = ["Street", "LandContour", "Utilities", "LandSlope", "Condition2", "RoofMatl", "BsmtCond", "BsmtFinType2", "Heating", "CentralAir", "Electrical", "Functional", "GarageQual", "GarageCond", "PavedDrive"]
df_train_categ.drop(cols_to_drop, axis=1, inplace=True)
df_test_categ.drop(cols_to_drop, axis=1, inplace=True)

# 删除缺失值比例超过30%的列
column_with_nan = df_train_categ.columns[df_train_categ.isnull().any()]
large_na = [col for col in column_with_nan if (df_train_categ[col].isna().sum() / df_train_categ.shape[0]) > 0.3]
df_train_categ.drop(large_na, axis=1, inplace=True)
df_test_categ.drop(large_na, axis=1, inplace=True)

# 使用最频繁的类别进行插补
categ_fill_null = {
    "GarageType": df_train_categ["GarageType"].mode().iloc[0],
    "GarageFinish": df_train_categ["GarageFinish"].mode().iloc[0],
    "BsmtQual": df_train_categ["BsmtQual"].mode().iloc[0],
    "BsmtFinType1": df_train_categ["BsmtFinType1"].mode().iloc[0],
    "MSZoning": df_train_categ["MSZoning"].mode().iloc[0],
    "Exterior1st": df_train_categ["Exterior1st"].mode().iloc[0],
    "KitchenQual": df_train_categ["KitchenQual"].mode().iloc[0],
    "SaleType": df_train_categ["SaleType"].mode().iloc[0]
}
df_train_categ = df_train_categ.fillna(value=categ_fill_null)
df_test_categ = df_test_categ.fillna(value=categ_fill_null)

# 将类别特征转换为二元特征
df_train_categ.drop(["SalePrice"], axis=1, inplace=True)
df_train_dummies = pd.get_dummies(df_train_categ)
df_test_dummies = pd.get_dummies(df_test_categ)

# 对齐训练集和测试集的列
common_cols = df_train_dummies.columns.intersection(df_test_dummies.columns)
df_train_dummies = df_train_dummies[common_cols]
df_test_dummies = df_test_dummies[common_cols]

# 合并数值特征和类别特征
df_train_final = pd.concat([df_train_imputed, df_train_dummies], axis=1)
df_test_final = pd.concat([df_test_imputed, df_test_dummies], axis=1)

# 删除数据集中的异常值
outliers1 = df_train_final[(df_train_final["GrLivArea"] > 4000) & (df_train_final["SalePrice"] <= 300000)].index.tolist()
outliers2 = df_train_final[(df_train_final["TotalBsmtSF"] > 4000) & (df_train_final["SalePrice"] <= 200000)].index.tolist()
outliers3 = df_train_final[(df_train_final["BsmtFinSF1"] > 4000)].index.tolist()
outliers4 = df_train_final[(df_train_final["OpenPorchSF"] > 500) & (df_train_final["SalePrice"] <= 100000)].index.tolist()
outliers = list(set(outliers1 + outliers2 + outliers3 + outliers4))
df_train_final = df_train_final.drop(df_train_final.index[outliers])
df_train_final = df_train_final.reset_index(drop=True)

# 创建新特征，删除旧特征
df_train_final["AgeSinceConst"] = df_train_final["YearBuilt"].max() - df_train_final["YearBuilt"]
df_test_final["AgeSinceConst"] = df_test_final["YearBuilt"].max() - df_test_final["YearBuilt"]
df_train_final.drop(["YearBuilt"], axis=1, inplace=True)
df_test_final.drop(["YearBuilt"], axis=1, inplace=True)

df_train_final["AgeSinceRemod"] = df_train_final["YearRemodAdd"] - df_train_final["YearBuilt"]
df_test_final["AgeSinceRemod"] = df_test_final["YearRemodAdd"] - df_test_final["YearBuilt"]
df_train_final.drop(["YearRemodAdd"], axis=1, inplace=True)
df_test_final.drop(["YearRemodAdd"], axis=1, inplace=True)

# 特征和目标变量
X = df_train_final[[i for i in df_train_final.columns if i != "SalePrice"]]
y = df_train_final["SalePrice"]

# 确保目标变量 y 是一个 DataFrame
y = y.reset_index(drop=True)
y = pd.DataFrame(y, columns=["SalePrice"])

# 特征和目标变量拆分
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(df_test_final)

# 训练模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train.values.ravel())  # 注意：y_train 需要被展平

# 预测
y_val_pred = model.predict(X_val_scaled)
y_test_pred = model.predict(X_test_scaled)

# 评估模型
print("Validation RMSE:", np.sqrt(mean_squared_error(y_val, y_val_pred)))
print("Validation R^2:", r2_score(y_val, y_val_pred))

# 保存预测结果
# submission = pd.DataFrame({'Id': Id_test_list, 'SalePrice': np.expm1(y_test_pred)})
# submission.to_csv('submission.csv', index=False)
