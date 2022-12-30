import numpy as np
import pandas as pd

def fill_null(df_original, df_with_module):
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
    return df_with_module
