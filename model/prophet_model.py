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
import pandas as pd
from prophet import Prophet
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns

# 绘图设置（适用于mac）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

datapath = '*************\\data\\jiadian_time_series_data.xlsx'
output_dir = '*******************t\\jiadian'


class ProphetTrain(object):
    def __init__(self, holidays, is_load):
        self.data = None
        self.train = None
        self.test = None
        self.holidays = holidays
        self.params = {
            'holidays': self.holidays}
        self.best_params = {}
        self.forecast = None
        self.score = [np.inf, np.inf, np.inf]
        self.model = None
        self.best_model = None
        self.is_load = is_load

    def load_data(self):
        self.data = pd.read_excel(datapath, sheet_name='pingban')
        self.data.rename(columns={'data': 'y', 'time': 'ds'},inplace=True)
        self.data['ds'] = pd.to_datetime(self.data['ds'])

    def data_size(self):
        if self.data is None:
            self.load_data()
        return self.data.shape[0] if self.data is not None else 0

    def split_data(self):
        # train和test数据集划分
        test_size = 30
        if self.data is None:
            self.load_data()
        self.train, self.test = self.data[:len(self.data) - test_size], self.data[len(self.data) - test_size:]

    def run(self, use_params=False, retrain=False):
        if self.train is None or self.test is None:
            self.split_data()

        if retrain or self.model is None:
            if use_params:
                self.model = Prophet(**self.params)
            else:
                self.model = Prophet()
            self.model.fit(self.train)

    def predict(self, step):
        if self.model is None:
            self.run()
        self.forecast = self.model.predict(self.test[:step])
        mae = mean_absolute_error(self.test[:step]['y'], self.forecast['yhat'])
        print('MAE: %.4f' % mae)
        mape = mean_absolute_percentage_error(self.test[:step]['y'], self.forecast['yhat'])
        print('MAPE: %.4f' % mape)
        rmse = sqrt(mean_squared_error(self.test[:step]['y'], self.forecast['yhat']))
        print('RMSE: %.4f' % rmse)

        R2 = r2_score(self.test[:step]['y'], self.forecast['yhat'])
        print('R2: %.4f' % R2)

        n = len(self.test[:step]['y'])
        feat_p = 1
        if (n - feat_p - 1) == 0:
            adjusted_R2 = np.nan
        else:
            adjusted_R2 = 1 - ((1 - r2_score(self.test[:step]['y'], self.forecast['yhat'])) * (n - 1)) / (n - feat_p - 1)
        print('adjusted_R2: %.4f' % adjusted_R2)

        return mae, mape, rmse, R2, adjusted_R2

    def grid_search(self, save_result=True):
        print('begin grid search...')
        # changepoint_range = [i / 10 for i in range(3, 10)] # changepoint:模型趋势发生改变的特殊日期，changepoint_range决定了changepoint能出现在离当前时间最近的时间点，changepont_range越大，changepoint可以出现的距离现在越近
        # seasonality_mode = ['additive', 'multiplicative'] # 周期性影响因素的模型方式
        seasonality_prior_scale = [0.05, 0.1, 0.5, 1, 5, 10, 15] # 改变周期性影响因素的强度，值越大，周期性因素在预测值中占比越大
        holidays_prior_scale = [0.05, 0.1, 0.5, 1, 5, 10, 15] # 改变节假日影响因素的强度，值越大，节假日因素在预测值中占比越大

        # for sm in seasonality_mode:
        #     for cp in changepoint_range:
        score_list = []
        for sp in seasonality_prior_scale:
            for hp in holidays_prior_scale:
                params = {
                    # "seasonality_mode": sm,
                    # "changepoint_range": cp,
                    "seasonality_prior_scale": sp,
                    "holidays_prior_scale": hp,
                    "holidays": self.holidays
                }
                self.params = params
                self.run(True, True)
                score = self.predict(len(self.test))
                score_list.append(score[2])
                if self.score[2] > score[2]: # use rmse
                    self.score = score
                    self.best_params = params
                    self.best_model = self.model
        print('finish grid search, result as follows:')
        print(f'mae:{self.score[0]}\nmape:{self.score[1]}\nrmse:{self.score[2]}\nparams:{self.best_params}')

        # plot
        plt.clf()
        score_result = np.array(score_list).reshape(7,7)
        df = pd.DataFrame(score_result, index=holidays_prior_scale, columns=seasonality_prior_scale)
        sns.heatmap(df, annot=True, fmt='.4g')
        plt.xlabel('holidays_prior_scale')
        plt.ylabel('seasonality_prior_scale')
        plt.savefig(output_dir + '\\' + '参数热力图.png', bbox_inches='tight')
        plt.show()
        print('heatmap plot finish!')

    def step_predict(self,name):
        if self.model is None:
            self.run()

        step_list = [3, 7, 14, 30]
        for step in step_list:
            print('step=%d day的评估指标如下：' % step)
            score = self.predict(step)

        # plot
        plt.clf()
        plt.figure(figsize=(50, 5))
        self.model.plot(self.forecast)
        plt.plot(self.test[:step_list[-1]].ds, self.test[:step_list[-1]]['y'], c='green', label="预期值", marker="s")
        plt.savefig(output_dir + '\\' + name+'_预测结果图.png', bbox_inches='tight')
        plt.show()

        plt.clf()
        self.model.plot_components(self.forecast)
        plt.savefig(output_dir + '\\' + name + '_成分趋势图.png', bbox_inches='tight')

        writer = pd.ExcelWriter(output_dir + '\\' "predictions.xlsx")
        self.forecast['yhat'].to_excel(writer)
        writer._save()
        writer.close()

    def process(self):

        if not self.is_load:
            # without holidays version
            print('-------without holidays version------------')
            self.run(False, True)
            self.save_model('不考虑holidays效应')
            self.step_predict('不考虑holidays效应')
            # with holidays version
            print('-------with holidays version------------')
            self.run(True, True)
            self.save_model('考虑holidays效应')
            self.step_predict('考虑holidays效应')
            # best result from grid search version
            print('-------best result from grid search version------------')
            self.grid_search()
            self.model = self.best_model
            self.save_model('考虑holidays效应_网格调参')
            self.step_predict('考虑holidays效应_网格调参')
        else:
            self.load_model('考虑holidays效应_网格调参')
            self.step_predict('考虑holidays效应_网格调参')



    def save_model(self, name):
        with open(output_dir + '\\' + f'{name}.pkl', 'wb') as fp:
            pickle.dump(self, fp)

    def load_model(self, name):
        with open(output_dir + '\\' + f'{name}.pkl', 'rb') as fp:
            return pickle.load(fp)


if __name__ == '__main__':

    # define holidays
    festivals = pd.DataFrame({
                  'holiday': 'festival',
                  'ds': pd.to_datetime(['2023-06-22', '2023-08-22', '2023-09-29',
                                        '2014-01-01', '2014-02-10', '2024-02-14',
                                        '2024-03-08', '2014-04-04', '2014-05-01',
                                        '2024-05-12', '2024-06-10']),
                  'lower_window': -1,
                  'upper_window': 1,
                })

    sales = pd.DataFrame({
                  'holiday': 'sales',
                  'ds': pd.to_datetime(['2023-06-18', '2023-11-11', '2023-12-12',
                                        '2014-06-18']),
                  'lower_window': -1,
                  'upper_window': 1,
                })
    # 没有加上学校假期，因为holidays的数量过多会加重模型的过拟合，一般应当小于整体数据的30%
    # 没有调整yearly_seasonality, weekly_seasonality, daily_seasonality周期性参数，因为模型会根据历史数据的长度进行默认设置，一般不做调整
    holidays = pd.concat((festivals, sales))
    prophet = ProphetTrain(holidays,True)
    prophet.process()
    print('all finished!')


