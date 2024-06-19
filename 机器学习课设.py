import os
os.environ["LOKY_MAX_CPU_COUNT"] = "8"
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, roc_auc_score
from lightgbm import LGBMClassifier

# 读取数据
df = pd.read_csv(r'D:\大二下\机器学习项目\pythonProject\课设\train.csv')
test = pd.read_csv(r'D:\大二下\机器学习项目\pythonProject\课设\test.csv')

# 统计订阅情况的数量
print(df['subscribe'].value_counts())

# 分离数值型特征和分类特征
Nu_feature = list(df.select_dtypes(exclude=['object']).columns)
Ca_feature = list(df.select_dtypes(include=['object']).columns)

# 使用LabelEncoder对分类特征进行编码
lb = LabelEncoder()
cols = Ca_feature
for m in cols:
    if m != 'subscribe':  # 排除'subscribe'列，因为它不存在于测试集中
        df[m] = lb.fit_transform(df[m])
        if m in test.columns:  # 只对测试集中存在的列进行编码
            test[m] = lb.transform(test[m])

# 将订阅标签转换为数值型（0和1）
df['subscribe'] = df['subscribe'].replace(['no', 'yes'], [0, 1])

# 划分数据集为训练集和测试集
X = df.drop(columns=['id', 'subscribe'])
Y = df['subscribe']
test_ids = test['id']
test = test.drop(columns='id')

# 设置LightGBM模型的参数
gbm = LGBMClassifier(n_estimators=600, learning_rate=0.01, boosting_type='gbdt',
                     objective='binary',
                     max_depth=-1,
                     random_state=2022,
                     metric='auc')

# 使用KFold进行交叉验证，并计算AUC值
result1 = []
mean_score1 = 0
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=2022)
for train_index, test_index in kf.split(X):
    x_train, x_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
    gbm.fit(x_train, y_train)
    y_pred1 = gbm.predict_proba(x_test)[:, 1]
    print('验证集AUC:{}'.format(roc_auc_score(y_test, y_pred1)))
    mean_score1 += roc_auc_score(y_test, y_pred1) / n_folds
    y_pred_final1 = gbm.predict_proba(test)[:, 1]
    result1.append(y_pred_final1)

# 模型评估
print('mean 验证集auc:{}'.format(mean_score1))
cat_pre1 = sum(result1) / n_folds

# 创建提交文件
ret1 = pd.DataFrame(cat_pre1, columns=['subscribe'])
ret1['subscribe'] = np.where(ret1['subscribe'] > 0.5, 1, 0)
submission = pd.concat([test_ids.reset_index(drop=True), ret1], axis=1)
submission.columns = ['id', 'subscribe']
submission.to_csv(r'D:\大二下\机器学习项目\pythonProject\课设\submission.csv', index=False)

# 创建提交文件
ret1 = pd.DataFrame(cat_pre1, columns=['subscribe'])
ret1['subscribe'] = np.where(ret1['subscribe'] > 0.5, 1, 0)
submission = pd.concat([test_ids.reset_index(drop=True), ret1], axis=1)
submission.columns = ['id', 'subscribe']

# 检查输出目录是否存在，如果不存在则创建它
output_dir = r'D:\大二下\机器学习项目\pythonProject\课设'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 尝试写入文件，捕获可能的异常
try:
    submission.to_csv(os.path.join(output_dir, 'submission.csv'), index=False)
except PermissionError as e:
    print("无法写入文件，可能是由于权限不足。错误信息：", str(e))
    # 在这里你可以添加逻辑来处理权限问题，比如提示用户更换路径或提升权限
except Exception as e:
    print("写入文件时发生未知错误。错误信息：", str(e))
    # 在这里你可以添加逻辑来处理其他类型的异常