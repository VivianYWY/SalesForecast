from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf #ACF与PACF
from statsmodels.tsa.arima.model import ARIMA #ARIMA模型
from statsmodels.graphics.api import qqplot  #qq图
from scipy import stats
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error

import warnings
warnings.filterwarnings("ignore")


# 绘图设置（适用于mac）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


output_dir = '****************\\jiadian'

# 使用BIC矩阵计算p和q的值
def cal_pqValue_BIC(D_data, diff_num=0) -> List[float]:
    # 定阶
    pmax = int(len(D_data) / 10)  # 一般阶数不超过length/10
    qmax = int(len(D_data) / 10)  # 一般阶数不超过length/10
    bic_matrix = []  # BIC矩阵
    # 差分阶数
    diff_num = 2

    for p in range(pmax + 1):
        tmp = []
        for q in range(qmax + 1):
            try:
                tmp.append(ARIMA(D_data, order=(p, diff_num, q)).fit().bic)
            except Exception as e:
                print(e)
                tmp.append(None)
        bic_matrix.append(tmp)

    bic_matrix = pd.DataFrame(bic_matrix)  # 从中可以找出最小值
    p, q = bic_matrix.stack().idxmin()  # 先用stack展平，然后用idxmin找出最小值位置。
    print(u'BIC最小的p值和q值为：%s、%s' % (p, q))
    return p, q

def cal_pqValue(diff_data, method='AIC_BIC', diff_num=0) -> List[float]:
    p = 0.0
    q = 0.0
    if method == 'AIC_BIC': # 使用AIC和BIC准则定阶q和p的值(推荐)
        AIC = sm.tsa.stattools.arma_order_select_ic(diff_data, max_ar=4, max_ma=4, ic='aic')['aic_min_order']
        BIC = sm.tsa.stattools.arma_order_select_ic(diff_data, max_ar=4, max_ma=4, ic='bic')['bic_min_order']
        print('---AIC与BIC准则定阶---')
        print('the AIC is{}\nthe BIC is{}\n'.format(AIC, BIC), end='')
        p = BIC[0]
        q = BIC[1]
        print('使用AIC和BIC准则定阶p和q的值：\np值为{}\nq值为{}\n'.format(p, q))

    else: # 使用BIC矩阵计算p和q的值
        pq_result = cal_pqValue_BIC(diff_data, diff_num)
        p = pq_result[0]
        q = pq_result[1]
        print('使用BIC矩阵计算p和q的值：\np值为{}\nq值为{}\n'.format(p, q))

    return p, q

def single_model(train, test, p, q, diff_num, step_flag)-> None:

    print('单次预测模型---------------------------')

    model = ARIMA(train, order=(p, diff_num, q)).fit()

    # 预测
    step_list = []
    if step_flag == 'day':
        step_list = [3, 7, 14, 30]
    elif step_flag == 'week':
        step_list = [1, 2, 4]

    predictions = {}
    for step in step_list:
        output = model.forecast(step)
        predictions[step] = output
        print('step=%d %s的评估指标如下：' % (step, step_flag))
        print('MAE: %.4f' % mean_absolute_error(test[:step], output))
        print('MAPE: %.4f' % mean_absolute_percentage_error(test[:step], output))
        print('RMSE: %.4f' % sqrt(mean_squared_error(test[:step], output)))

        R2 = r2_score(test[:step], output)
        print('R2: %.4f' % R2)

        n = len(test[:step])
        feat_p = 1
        if (n - feat_p - 1) == 0:
            adjusted_R2 = np.nan
        else:
            adjusted_R2 = 1 - ((1 - r2_score(test[:step], output)) * (n - 1)) / (n - feat_p - 1)
        print('adjusted_R2: %.4f' % adjusted_R2)

    # plot
    plt.clf()
    train = pd.concat([train, test[:1]])
    plt.figure(figsize=(50, 5))
    plt.title("单次预测： step="+str(step_list[-1])+" "+step_flag, fontdict={'size': 20})
    plt.plot(train[len(train) - 60:].index, train[len(train) - 60:]['deal_data'], c='blue', label="部分历史数据", marker=".")
    plt.plot(test[:step_list[-1]].index, test[:step_list[-1]]['deal_data'], c='green', label="预期值", marker="s")
    plt.plot(test[:step_list[-1]].index, predictions[step_list[-1]], c='red', label="预测值", marker="*")

    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel("时间", fontdict={'size': 16})
    plt.ylabel("销量", fontdict={'size': 16})
    plt.tick_params(axis='x', labelrotation=90)
    plt.savefig(output_dir + '\\' + '单次预测模型预测结果.png', bbox_inches='tight')
    plt.show()

    writer = pd.ExcelWriter(output_dir + '\\' "predictions.xlsx")
    predictions[step_list[-1]].to_excel(writer)
    writer._save()
    writer.close()


def rolling_model(train, test, p, q, diff_num, step_flag)-> None:

    print('滚动预测模型---------------------------')

    step = 1
    if step_flag == 'day':
        step = 3
    elif step_flag == 'week':
        step = 1

    predictions = []
    for t in range(0,len(test),step):
        model = ARIMA(train, order=(p, diff_num, q)).fit()
        output = model.forecast(step)
        predictions.extend(output.tolist())
        obs = test[t:t+step]
        train = pd.concat([train,obs])

    print('MAE: %.4f' % mean_absolute_error(test, predictions))
    print('MAPE: %.4f' % mean_absolute_percentage_error(test, predictions))
    print('RMSE: %.4f' % sqrt(mean_squared_error(test, predictions)))

    R2 = r2_score(test, predictions)
    print('R2: %.4f' % R2)

    n = len(predictions)
    feat_p = 1
    if (n - feat_p - 1) == 0:
        adjusted_R2 = np.nan
    else:
        adjusted_R2 = 1-((1-r2_score(test,predictions))*(n-1))/(n-feat_p-1)
    print('adjusted_R2: %.4f' % adjusted_R2)

    # plot
    plt.clf()
    plt.figure(figsize=(50, 5))
    plt.title("滚动预测： step="+str(step)+" "+step_flag, fontdict={'size': 20})
    plt.plot(train[len(train)-len(test)+1-60:len(train)-len(test)+1].index, train[len(train)-len(test)+1-60:len(train)-len(test)+1]['deal_data'], c='blue', label="部分历史数据", marker=".")
    plt.plot(test.index, test['deal_data'], c='green', label="预期值", marker="s")
    plt.plot(test.index, predictions, label="预测值", c='red', marker="*")

    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel("时间", fontdict={'size': 16})
    plt.ylabel("销量", fontdict={'size': 16})
    plt.tick_params(axis='x', labelrotation=90)
    plt.savefig(output_dir + '\\' + '滚动预测模型预测结果.png', bbox_inches='tight')
    plt.show()


# 计算时序序列模型
def cal_time_series(train, test, step_flag) -> None:

    # 原始数据分析 -------------------------------------
    # 绘制时序图
    plt.figure(figsize=(50, 5))
    plt.plot(train.index, train['deal_data'], c='blue', label="原始序列", marker=".")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel("时间", fontdict={'size': 16})
    plt.ylabel("销量", fontdict={'size': 16})
    plt.tick_params(axis='x', labelrotation=90)
    plt.savefig(output_dir + '\\' + '原始序列时序图.png', bbox_inches='tight')
    plt.show()

    # 绘制自相关图
    plt.clf()
    plt.tick_params(axis='x', labelrotation=90)
    plt.figure(figsize=(50, 5))
    plot_acf(train)
    plt.savefig(output_dir + '\\' + '原始序列自相关图.png', bbox_inches='tight')
    plt.show()

    # 绘制偏自相关图
    plt.clf()
    plt.tick_params(axis='x', labelrotation=90)
    plt.figure(figsize=(50, 5))
    plot_pacf(train)
    plt.savefig(output_dir + '\\' + '原始序列偏自相关图.png', bbox_inches='tight')
    plt.show()

    # 时序数据平稳性检测
    original_ADF = ADF(train[u'deal_data'])
    print(u'原始序列的ADF检验结果为：', original_ADF)

    # 对数序数据进行d阶差分运算，化为平稳时间序列 ------------------------------------------------
    diff_num = 0 # 差分阶数
    diff_data = train     # 差分数序数据
    ADF_p_value = ADF(train[u'deal_data'])[1]
    n = 0
    while ADF_p_value > 0.01:
    #while n < 2:
        diff_data = diff_data.diff(periods=1).dropna()
        diff_num = diff_num + 1
        ADF_result = ADF(diff_data[u'deal_data'])
        ADF_p_value = ADF_result[1]
        print("ADF_p_value:{ADF_p_value}".format(ADF_p_value=ADF_p_value))
        print(u'{diff_num}差分的ADF检验结果为：'.format(diff_num = diff_num), ADF_result )

        # 绘制时序图
        plt.clf()
        plt.figure(figsize=(50, 5))
        plt.plot(diff_data.index, diff_data['deal_data'], c='blue', label=str(diff_num)+"差分序列", marker=".")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xlabel("时间", fontdict={'size': 16})
        plt.ylabel("销量", fontdict={'size': 16})
        plt.tick_params(axis='x', labelrotation=90)
        plt.savefig(output_dir + '\\' + str(diff_num)+'差分序列时序图.png', bbox_inches='tight')
        plt.show()

        # 绘制自相关图
        plt.clf()
        plt.figure(figsize=(50, 5))
        plt.tick_params(axis='x', labelrotation=90)
        plot_acf(diff_data)
        plt.savefig(output_dir + '\\' + str(diff_num)+'差分序列自相关图.png', bbox_inches='tight')
        plt.show()

        # 绘制偏自相关图
        plt.clf()
        plt.figure(figsize=(50, 5))
        plt.tick_params(axis='x', labelrotation=90)
        plot_pacf(diff_data)
        plt.savefig(output_dir + '\\' + str(diff_num)+'差分序列偏自相关图.png', bbox_inches='tight')
        plt.show()

        n = n + 1

    # 白噪声检测
    print(u'最终差分序列的白噪声检验结果为：', acorr_ljungbox(diff_data, lags=1))  # 返回统计量和p值
    # p,q值获取
    p, q = cal_pqValue(diff_data,'AIC_BIC', diff_num)

    # 构建时间序列模型
    # 单次预测模型
    single_model(train, test, p, q, diff_num, step_flag)
    # 滚动预测模型
    rolling_model(train, test, p, q, diff_num, step_flag)

    print('finish!')


if __name__ == '__main__':

    mode = ''

    # data
    if mode == 'test':
        # 数据测试:
        need_data = {'2016-02': 44964.03, '2016-03': 56825.51, '2016-04': 49161.98, '2016-05': 45859.35,
                     '2016-06': 45098.56,
                     '2016-07': 45522.17, '2016-08': 57133.18, '2016-09': 49037.29, '2016-10': 43157.36,
                     '2016-11': 48333.17,
                     '2016-12': 22900.94,
                     '2017-01': 67057.29, '2017-02': 49985.29, '2017-03': 49771.47, '2017-04': 35757.0, '2017-05': 42914.27,
                     '2017-06': 44507.03, '2017-07': 40596.51, '2017-08': 52111.75, '2017-09': 49711.18,
                     '2017-10': 45766.09,
                     '2017-11': 45273.82, '2017-12': 22469.57,
                     '2018-01': 71032.23, '2018-02': 37874.38, '2018-03': 44312.24, '2018-04': 39742.02,
                     '2018-05': 43118.38,
                     '2018-06': 33859.69, '2018-07': 38910.89, '2018-08': 39138.42, '2018-09': 37175.03,
                     '2018-10': 44159.96,
                     '2018-11': 46321.72, '2018-12': 22410.88,
                     '2019-01': 61241.94, '2019-02': 31698.6, '2019-03': 44170.62, '2019-04': 47627.13, '2019-05': 54407.37,
                     '2019-06': 50231.68, '2019-07': 61010.29, '2019-08': 59782.19, '2019-09': 57245.15,
                     '2019-10': 61162.55,
                     '2019-11': 52398.25, '2019-12': 15482.64,
                     '2020-01': 38383.97, '2020-02': 26943.55, '2020-03': 57200.32, '2020-04': 49449.95,
                     '2020-05': 47009.84,
                     '2020-06': 49946.25, '2020-07': 56383.23, '2020-08': 60651.07}
        data = {'time_data': list(need_data.keys()), 'deal_data': list(need_data.values())}
        df = pd.DataFrame(data)
        df.set_index(['time_data'], inplace=True)  # 设置索引

    else:
        # 正式数据（从excel中读取）:
        path = 'C:\\Users\\Vivia\\工作\\code\\SalesForecast\\data\\jiadian_time_series_data.xlsx'
        df = pd.read_excel(path,sheet_name='pingban')
        df.rename(columns={'data': 'deal_data', 'time': 'time_data'}, inplace=True)
        df.set_index(['time_data'], inplace=True)  # 设置索引

    # train和test数据集划分
    test_size = 30
    train, test = df[:len(df) - test_size], df[len(df) - test_size:]
    step_flag = 'day'
    cal_time_series(train, test, step_flag) # 模型调用