import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm

def Corr(df_wo_module):
    corr_matrix = df_wo_module.corr()
    os.makedirs('./Analysis/train_data', exist_ok=True)
    with open(os.path.join('./Analysis/train_data', "correlation_matrix.txt"), 'w') as f:
        f.write(corr_matrix.to_string())
    return corr_matrix

def module_performance_analysis(df_with_module, savedir):
    os.makedirs(savedir, exist_ok=True)
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
    plt.savefig(os.path.join(savedir, "generation_distribute_interval-{}".format(interval)))
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
    plt.savefig(os.path.join(savedir, "avg_gen-irr_interval"))
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
    plt.savefig(os.path.join(savedir, "avg_gen(persent)-irr_interval"))
    plt.close()
    return df_with_module

def Lag_data_EDA(df, featurename):
    input = df[featurename]
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(input.values.squeeze(), lags=200, ax=ax1)
    ax2 = fig.add_subplot(212)
    # fig.title('par')
    fig = sm.graphics.tsa.plot_acf(input, lags=200, ax=ax2)
    plt.savefig('./Analysis/train_data/auto_corr.png')
    plt.close()
    return df