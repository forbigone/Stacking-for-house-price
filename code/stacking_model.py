import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder


''' 
导入数据
'''
file = os.path.abspath(os.path.join(os.getcwd(), ".."))  # 返回代码所在上一级目录
data_file = os.path.join(file, 'data/train.csv')  # 返回训练数据所在文件位置
train_df = pd.read_csv(data_file)
data_file = os.path.join(file, 'data/test.csv')  # 返回测试数据所在文件位置
test_df = pd.read_csv(data_file)



'''
数据处理和特征分析
'''
# 保存id
train_ID = train_df['Id']
test_ID = test_df['Id']

# 删除id
train_df.drop("Id", axis = 1, inplace = True)
test_df.drop("Id", axis = 1, inplace = True)

# 去掉面积4000以上，但其价格不到200000的异常值
train_df.drop(train_df[(train_df['GrLivArea']>4000)&(train_df['GrLivArea']<30000)].index,inplace=True)

         

# 0. 对数变换
train_df['SalePrice_Log1p'] = np.log1p(train_df['SalePrice'])
                 
del train_df["SalePrice"]

# 将测试数据和训练数据联合一起进行特征分析
size_train_df = train_df.shape[0]
size_test_df = test_df.shape[0]
target_variable = train_df['SalePrice_Log1p'].values
data = pd.concat((train_df, test_df),sort=False).reset_index(drop=True)
data.drop(['SalePrice_Log1p'], axis=1, inplace=True)







''' 
缺失值处理 
'''
data.count().sort_values().head()  # 统计不为空的值，反映出有缺失值的数据

# 统计每个字段的缺失值数量，绘制条形图
data_na = (data.isnull().sum() / len(data)) * 100
data_na.drop(data_na[data_na==0].index,inplace=True)
data_na = data_na.sort_values(ascending=False)


# 填充nil
features_fill_na_none = ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu',
               'GarageQual','GarageCond','GarageFinish','GarageType',
               'BsmtExposure','BsmtCond','BsmtQual','BsmtFinType1','BsmtFinType2',
               'MasVnrType']

# 填充0
features_fill_na_0 = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'MasVnrArea',
                      'BsmtFullBath','BsmtHalfBath', 'BsmtFinSF1', 'BsmtFinSF2', 
                      'BsmtUnfSF', 'TotalBsmtSF']

# 填众数
features_fill_na_mode = ["Functional", "MSZoning", "SaleType", "Electrical", 
                         "KitchenQual", "Exterior2nd", "Exterior1st"]

for feature_none in features_fill_na_none:
    data[feature_none].fillna('None',inplace=True)
    
for feature_0 in features_fill_na_0:
    data[feature_0].fillna(0,inplace=True)

for feature_mode in features_fill_na_mode:
    mode_value = data[feature_mode].value_counts().sort_values(ascending=False).index[0]  # 排序取值最大的-众数
    data[feature_mode] = data[feature_mode].fillna(mode_value)

# 用中值代替
#   df.groupby按Neighborhood字段进行分组，然后对每个分组的LotFrontage求中值，填充缺失值  
data["LotFrontage"] = data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

# 像 Utilities 这种总共才两个值，同时有一个值是作为主要的，这种字段是无意义的，应该删除
data.drop(['Utilities'], axis=1,inplace=True)


# 类型转换      将某些类别类型但用数字表示的强制转换成文本
#MSSubClass=The building class
data['MSSubClass'] = data['MSSubClass'].apply(str)
#Changing OverallCond into a categorical variable
data['OverallCond'] = data['OverallCond'].astype(str)
#Year and month sold are transformed into categorical features.
data['YrSold'] = data['YrSold'].astype(str)
data['MoSold'] = data['MoSold'].astype(str)






'''
特征工程
'''
# 对类别数据进行LabelEncoder编码
encode_cat_variables = ('Alley', 'BldgType', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'CentralAir', 
                        'Condition1', 'Condition2', 'Electrical', 'ExterCond', 'ExterQual', 'Exterior1st', 'Exterior2nd', 'Fence', 
                        'FireplaceQu', 'Foundation', 'Functional', 'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType', 
                        'Heating', 'HeatingQC', 'HouseStyle', 'KitchenQual', 'LandContour', 'LandSlope', 'LotConfig', 'LotShape', 
                        'MSSubClass', 'MSZoning', 'MasVnrType', 'MiscFeature', 'MoSold', 'Neighborhood', 'OverallCond', 'PavedDrive', 
                        'PoolQC', 'RoofMatl', 'RoofStyle', 'SaleCondition', 'SaleType', 'Street', 'YrSold')
# 剩下的数值特征进行正态分布装换
numerical_features = [col for col in data.columns if col not in encode_cat_variables]
print("Categorical Features: %d"%len(encode_cat_variables))  # 46
print("Numerical Features: %d"%len(numerical_features))  # 32

# for variable in encode_cat_variables:
#     lbl = LabelEncoder() 
#     lbl.fit(list(data[variable].values)) 
#     data[variable] = lbl.transform(list(data[variable].values))

for variable in data.columns:
    if variable not in encode_cat_variables:
        data[variable] = data[variable].apply(float)
    else:
        data[variable] = data[variable].apply(str)

print(data.shape)
# one-hot encode编码，get_dummies将字符字段进行独热编码
data = pd.get_dummies(data)
print(data.shape)

# 可以计算一个总面积指标，作为新字段
data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']




# box-cox变换        一种广义幂变换方法，用于连续的响应变量不满足正态分布的情况
# 计算数值型变量的偏态
skewed_features = data[numerical_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
# 提取偏态绝对值大于0.75的字段
skewed_features = skewed_features[abs(skewed_features) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewed_features.shape[0]))  # 20
# box-cox变换降低量的偏态
skewed_features_name = skewed_features.index
lam = 0.15 # 超参数
for feat in skewed_features_name:
    tranformer_feat = boxcox1p(data[feat], lam)
    data[feat] = tranformer_feat

data[numerical_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)




# 将训练集和测试集分开
train = data[:size_train_df]
test = data[size_train_df:]





'''
建模
'''
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

# 定义一个交叉评估函数 Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, target_variable, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
    
    
# LASSO回归(LASSO Regression)             Lasso score: 0.1101 (0.0058)
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    
# 岭回归（Kernel Ridge Regression）       Lasso score: 0.1152 (0.0043)
KRR = make_pipeline(RobustScaler(), KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5))
score = rmsle_cv(KRR)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))    

# 弹性网络回归(Elastic Net Regression)    Lasso score: 0.1100 (0.0059)
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
score = rmsle_cv(ENet)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))   
 
# 提升树(Gradient Boosting Regression):    Lasso score: 0.1180 (0.0088)
GBoost = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state=5)

score = rmsle_cv(GBoost)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std())) 
   
# XGBoost                               Lasso score: 0.1208 (0.0035)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=300,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

score = rmsle_cv(model_xgb)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    
# LightGBM                      Lasso score: 0.1176 (0.0058)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=300,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
score = rmsle_cv(model_lgb)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))    
    
  
    


'''
集成学习
'''
# 平均模型         score: 0.1083 (0.0061)
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models, list_weight = [], **kwargs):
        self.models = models
        # 计算归一化权重
        if list_weight == []:
            list_weight = [1] * len(models)
        sum_ = sum(list_weight)
        for i in range(len(list_weight)):
            list_weight[i] /= sum_
        self.list_weight = list_weight
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            self.models_[i].predict(X)*self.list_weight[i] for i in range(len(self.models_))
        ])
        return np.sum(predictions, axis=1)

averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso), list_weight=[0.1100,0.1180,0.1152,0.1101])

score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



# 堆叠模型(Stacking Averaged Models)        score: 0.1088 (0.0062)
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        
   
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]  #  4×5 list 存放训练好的模型
        self.meta_model_ = clone(self.meta_model) # 复制基准模型，因为这里会有多个模型
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # 训练基准模型，基于基准模型训练的结果导出成特征
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y): #分为预测holdout_index和训练train_index
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # 将基准模型预测数据作为特征用来给meta_model训练
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
    
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)

from sklearn.linear_model import LinearRegression
meta_model = KRR
stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR, lasso),
                                                 meta_model = meta_model,
                                                n_folds=10)
score = rmsle_cv(stacked_averaged_models)
print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
 



import random
from numpy import median
def subsample(dataset_x, ratio_sample, ratio_feature, i, list_fearure):   # 创建数据集的随机子样本
    """random_forest(评估算法性能，返回模型得分)
    Args:
        dataset         训练数据集
        ratio           训练数据集的样本比例,特征比例
    Returns:
        sample          随机抽样的训练样本序列号
        test_list       随机抽样后的剩下的测试样本序列号
        feature         随机抽样的特征序列号
    """
    random.seed(i)  # 固定随机值
    sample = list()
    # 训练样本的按比例抽样。
    # round() 方法返回浮点数x的四舍五入值。
    n_sample = round(len(dataset_x) * ratio_sample)
    n_feature = round(dataset_x.shape[1] * ratio_feature)
    sample = random.sample(range(len(dataset_x)), n_sample)
    feature = np.random.choice(a=range(dataset_x.shape[1]), size=n_feature, replace=False, p=list_fearure)
    test_list = list(set(range(len(dataset_x))) - set(sample))

    return sample, test_list, feature


# RF堆叠模型(RF Stacking Averaged Models)        score: 0.1141 (0.0073)
class RfStackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_tree=20, ratio_sample=1, ratio_feature=1,list_fearure=[]):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_tree = n_tree
        self.ratio_sample = ratio_sample
        self.ratio_feature = ratio_feature
        self.list_fearure = list_fearure

   
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]  #  4×5 list 存放训练好的模型
        self.meta_model_ = clone(self.meta_model) # 复制基准模型，因为这里会有多个模型
        self.list_weight = [list() for x in self.base_models]  #  4×20 list 存放预测的权重
        self.list_feature = [list() for x in self.base_models]  #  4×20 list 存放特征
        
        n_tree = self.n_tree
        ratio_sample = self.ratio_sample
        ratio_feature = self.ratio_feature
        list_fearure = self.list_fearure
        # 训练基准模型，基于基准模型训练的结果导出成特征
        # that are needed to train the cloned meta-model
#        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        rf_predictions = [list() for x in self.base_models]
        rf_y = []
        rf_x0 = []
        rf_x1 = []
        
#        for i in range(X.shape[0]):
#            out_of_fold_predictions[i, 4] = X[i,1]
        for i, model in enumerate(self.base_models):
            out__predictions = np.zeros((X.shape[0], n_tree))
            for j in range(n_tree):
                train_list, test_list, feature = subsample(X, ratio_sample, ratio_feature, j, list_fearure)
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[np.ix_(train_list, feature)], y[train_list])
                y_pred = instance.predict(X[np.ix_(test_list, feature)])
                # list(set(feature)|set(list_fearure))
                rf_predictions[i].extend(y_pred)
                if i == 0:
                    rf_x0.extend(test_list)
                    rf_x1.extend(feature)
                    rf_y.extend(y[test_list])
                mse = mean_squared_error(y_pred, y[test_list])
                self.list_weight[i].append(mse)
                out__predictions[:, j] = instance.predict(X[np.ix_(range(X.shape[0]), feature)])
                self.list_feature[i].append(feature) 
                                        
            # 权重计算
            sum_weight = sum(self.list_weight[i])
            num_weight = len(self.list_weight[i])
            mid_weight = median(self.list_weight[i])
            for j in range(num_weight):
                self.list_weight[i][j] = (sum_weight - self.list_weight[i][j]) / sum_weight / (num_weight-1)
               
        # 将基准模型预测数据作为特征用来给meta_model训练
        rf_x = pd.DataFrame(rf_predictions)
        rf_x = rf_x.T
        self.meta_model_.fit(rf_x, rf_y)

        return self
    
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X[np.ix_(range(X.shape[0]), self.list_feature[i][j])])*self.list_weight[i][j] for j, model in enumerate(base_models)]).sum(axis=1)
            for i, base_models in enumerate(self.base_models_) ])
#        meta_features = np.c_[meta_features,X[:,1]]
        return self.meta_model_.predict(meta_features)



GBoost.fit(train,target_variable)
list_fearure = GBoost.feature_importances_
list_non = list(np.nonzero(list_fearure)[0])
meta_model = GBoost
rf_stacked_averaged_models = RfStackingAveragedModels(base_models = (ENet, KRR, lasso),
                                                 meta_model = meta_model,
                                                n_tree=20, ratio_sample=0.8, 
                                                ratio_feature=0.6, 
                                                list_fearure=list_fearure)

score = rmsle_cv(rf_stacked_averaged_models)
print("RF Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))


'''    
y_train = target_variable
x_train = train.values   
rf_stacked_averaged_models.fit(x_train,y_train)
y = rf_stacked_averaged_models.predict(test.values)
y = np.expm1(y)  # log1p逆变换
df = pd.DataFrame({'Id':test_ID,'SalePrice':y})
path1 = file + '/code/fianl1.csv'
df.to_csv(path1, encoding = 'utf-8', index = False)  # 去掉index，保留头部
'''


'''
导出结果
'''




