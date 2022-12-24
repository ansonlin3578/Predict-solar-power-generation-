import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error

class load_data(object):
    def load_train(path, dataset):
        df = pd.read_csv(path)
        dataset["train"]["data"] = df
        dataset["train"]["label"] = df["Generation"]
    def load_test(path, dataset):
        df = pd.read_csv(path)
        dataset["test"]["data"] = df
        dataset["test"]["label"] = df["Generation"]

dataset = {
            "train" : {
                        "data" : [],
                        "label" : []
                        },
            "test" : {
                        "data" : [],
                        "label" : []
                        }
            }

load_data.load_train(r"./dataset/train.csv", dataset)
load_data.load_test(r"./dataset/test.csv", dataset)
# print(dataset["train"]["data"])
# print(dataset["train"]["label"])

df_original = dataset["train"]["data"]
df_with_module = df_original[["Generation", "Module", "Irradiance", "Capacity", "Irradiance_m", "Temp"]]
##calculate the correlation matrix and filts the columns which have value lower than 0.1 with "Generation"
df_wo_module = df_original.drop(columns=["Module"])

corr_matrix = df_wo_module.corr()
to_filter = corr_matrix["Generation"]
for index in to_filter.index:
    if abs(to_filter[index]) < 0.1:
        df_wo_module = df_wo_module.drop(columns=[index])
dataset["train"]["data"] = df_wo_module

################################# fill the nan value ####################################

fill_temp = df_original.iloc[:, 10].values
nan_check = np.isnan(fill_temp)     
for i in range(len(fill_temp)):     #fill temp value of training data(前6天 & 最近的後6天 取averagee)
    stack = []
    if nan_check[i]:
        left_sum = np.sum(fill_temp[i-6 : i])   
        j = i+1
        while len(stack) < 6:
            if not nan_check[j]:
                stack.append(fill_temp[j])
            j += 1
        right_sum = sum(stack)
        avg = (left_sum + right_sum)/12
        fill_temp[i] = avg

df_with_module["Temp"] = fill_temp
print("df_with_module:")        
for column in df_with_module:   #確認每個column的nan value數量
    print("nan value in {} : ".format(column) , df_with_module[column].isna().sum())
    if column == "Irradiance_m":
        print("'0' nan value in {} : ".format(column) , (df_with_module[column] == 0).sum())

fill_irr = df_original.iloc[:, 4].values    #fill irradiance of training data
nan_check = np.isnan(fill_irr)              #依照temp來填irradiance (已經先確認過module的發電效率高低，才開使填補nan)
for i in range(len(fill_irr)):    #irr[i] = irr[i-3] + 3*irr[i-3]*((temp[i+3] = temp[i-3]) / temp[i-3])
    if nan_check[i]:
        print(fill_irr[i])
        if (i-3 >= 0) and (i+3 < len(fill_irr)):
            ans = fill_irr[i-3] + 3 * fill_irr[i-3] * ((fill_temp[i+3] - fill_temp[i-3]) / fill_temp[i-3])
            if ans < 0:
                ans = min(fill_irr)
            fill_irr[i] = ans
df_with_module["Irradiance"] = fill_irr
print("df_with_module:")
for column in df_with_module:   #確認每個column的nan value數量
    print("nan value in {} : ".format(column) , df_with_module[column].isna().sum())
    if column == "Irradiance_m":
        print("'0' nan value in {} : ".format(column) , (df_with_module[column] == 0).sum())

################################# training data preprcessing ####################################
df_with_module = df_with_module.sort_values(by=["Generation"])
interval = 20
df_with_module["grade"] = df_with_module["Generation"] // interval    
df_intervals = df_with_module.groupby("grade")
grades_count = []
grade_interval = []
for gp_by_interval in df_intervals:     #發電量以"20"為interval，畫出所有資料的發電量分布
    count_each_interval = len(gp_by_interval[1])
    grades_count.append(count_each_interval)
    grade_interval.append(gp_by_interval[0])

test = plt.bar(x=grade_interval, height=grades_count, snap=True, color="r")
plt.xlabel("generation interval")
plt.ylabel("numbers of each interval")
plt.savefig("./generation distribute interval-{}".format(interval))
plt.close()


df_with_highGen = df_with_module.loc[(df_with_module["Generation"] <= 175*20)]  #剔除outlier
print("nan value in Irradiance : " , df_with_highGen["Irradiance"].isna().sum()) #計算nan value of irradiance
df_with_irr = df_with_highGen.sort_values(by=["Irradiance"])
df_with_irr["irr_interval"] = np.nan
df_with_irr["irr_interval"] = df_with_irr["Irradiance"] // 2    
# print(df_with_irr)
gp_by_irr_interval = df_with_irr.groupby("irr_interval")     
# print(gp_by_irr_interval.size())

module_count = [0, 0, 0, 0]
avg_gen_each_module = [0, 0, 0, 0]
percent_each_module = [0, 0, 0, 0]
module_name = ["AUO PM060MW3 320W", "MM60-6RT-300", "SEC-6M-60A-295", "AUO PM060MW3 325W"]
plot_idx = 1
for interval, df in gp_by_irr_interval:     #計算每個irradiance interval(2)下，每個module的平均發電量，為了瞭解哪個module發電效率較高
    gp_md = df.groupby("Module")            #，作為categories encoding的依據
    for md, md_df in gp_md:
        if md == "AUO PM060MW3 320W":
            avg_gen_each_module[0] = md_df["Generation"].mean()
        elif md == "MM60-6RT-300":
            avg_gen_each_module[1] = md_df["Generation"].mean()
        elif md == "SEC-6M-60A-295":
            avg_gen_each_module[2] = md_df["Generation"].mean()
        elif md == "AUO PM060MW3 325W":
            avg_gen_each_module[3] = md_df["Generation"].mean()

    plt.subplot(3, 5, plot_idx)
    img_interval = plt.bar(x=module_name, height=avg_gen_each_module)
    plt.title("interval {} ~ {}".format(interval*2 , (interval+1)*2))
    plt.xlabel("modules name")
    plt.ylabel("avg_gen")
    plt.ylim(0,3000)
    plot_idx += 1
    avg_gen_each_module = [0, 0, 0, 0]
plt.savefig("avg_gen-irr_interval")
plt.close()

plot_idx = 1
for interval, df in gp_by_irr_interval:
    gp_md = df.groupby("Module")
    for md, md_df in gp_md:
        if md == "AUO PM060MW3 320W":
            avg_gen_each_module[0] = md_df["Generation"].mean()
        elif md == "MM60-6RT-300":
            avg_gen_each_module[1] = md_df["Generation"].mean()
        elif md == "SEC-6M-60A-295":
            avg_gen_each_module[2] = md_df["Generation"].mean()
        elif md == "AUO PM060MW3 325W":
            avg_gen_each_module[3] = md_df["Generation"].mean()

    for i in range(4):                          #同上，只是換算成百分比
        percent_each_module[i] = avg_gen_each_module[i] / sum(avg_gen_each_module) *100
    plt.subplot(3, 5, plot_idx)
    img_interval = plt.bar(x=module_name, height=percent_each_module)
    plt.title("interval {} ~ {}".format(interval*2 , (interval+1)*2))
    plt.xlabel("modules name")
    plt.ylabel("avg_gen")
    plt.ylim(0,100)
    plot_idx += 1
    percent_each_module = [0, 0, 0, 0]
    avg_gen_each_module = [0, 0, 0, 0]
plt.savefig("avg_gen(persent)-irr_interval")
plt.close()
######################################## Module encoding #################################################
only_module = df_with_module[["Module"]]
only_module = pd.concat((only_module, pd.get_dummies(only_module.Module)), 1)
only_module = only_module.drop(["Module"], axis=1)
only_module["MM60-6RT-300"] = 1.5 * only_module["MM60-6RT-300"] #onehot encodeing後，將module("MM60-6RT-300")的scale * 1.5

onehot_df = pd.concat(objs=(df_with_module, only_module), axis=1)
# for column in onehot_df:
#     print("nan value in {} : ".format(column) , onehot_df[column].isna().sum())

######################################## create model  #################################################
train_label = onehot_df.Generation.values   #整理成最後要丟進去model的資料型態
train_data = onehot_df.drop(columns=["Generation", "Module", "grade"])
n_folds = 5
def rmsle_cv(X, y, model, n_folds):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X.values)
    rmse= np.sqrt(-cross_val_score(model, X.values, y, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

def simple_model_test(X, y ,n_folds):
    model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, learning_rate=0.05, max_depth=3, min_child_weight=1.7817, n_estimators=2200,
                                 reg_alpha=0.4640, reg_lambda=0.8571, subsample=0.5213, random_state =7, nthread = -1)
    model_list = {'XGB':model_xgb}
    for n,model in model_list.items():
        score = rmsle_cv(X, y, model, n_folds)
        print('\n{} score: {:.4f} ({:.4f})\n'.format(n, score.mean(), score.std()))
    return model_xgb

model_xgb = simple_model_test(train_data, train_label ,n_folds)
###################################### test preprocessing #################################################
df_test_ori = dataset["test"]["data"]
df_test_to_pre = df_test_ori.drop(columns = ["ID", "Temp_m", "Date", "Generation", "Lat", "Lon", "Angle"])

temp_list = df_test_to_pre.Temp.values
nan_check = np.isnan(temp_list)
for i in range(len(temp_list)):     #填補test.csv中，"Temp" column 的 nan value，方法與train.csv相同
    stack = []
    if nan_check[i]:
        left_sum = np.sum(temp_list[i-6 : i])
        j = i+1
        while len(stack) < 6:
            if not nan_check[j]:
                stack.append(temp_list[j])
            j += 1
        right_sum = sum(stack)
        avg = (left_sum + right_sum)/12
        temp_list[i] = avg
df_test_to_pre["Temp"] = temp_list

only_module = df_test_to_pre[["Module"]]    #onehot encodeing後，將module("MM60-6RT-300")的scale * 1.5
only_module = pd.concat((only_module, pd.get_dummies(only_module.Module)), 1)
only_module = only_module.drop(["Module"], axis=1)
only_module["MM60-6RT-300"] = 1.5 * only_module["MM60-6RT-300"]

test_onehot_df = pd.concat(objs=(df_test_to_pre, only_module), axis=1)
# for column in test_onehot_df:
#     print("nan value in {} : ".format(column) , onehot_df[column].isna().sum())
test_data = test_onehot_df.drop(columns=["Module"])

##################################### test prediction ###############################################
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

model_xgb.fit(train_data, train_label)
xgb_train_pred = model_xgb.predict(train_data)
xgb_pred = model_xgb.predict(test_data)
print(rmsle(train_label, xgb_train_pred))

