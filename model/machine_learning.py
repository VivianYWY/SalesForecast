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
from sklearn.preprocessing import StandardScaler
import pandas as pd
from prophet import Prophet
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
from lightgbm import LGBMRegressor
from xgboost import plot_importance as plot_importance_xgb
from lightgbm import plot_importance as plot_importance_gbm
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
import shap

# 绘图设置（适用于mac）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['figure.autolayout'] = True

datapath = '*****************\\spu粒度（天）_features.xlsx'
output_dir = '**********************\\meizhuang'


class MLTrain(object):
    def __init__(self, flag, method, is_val, scene,feat):
        self.feat = feat
        if scene == 'jiadian':
            self.mappings = {'耳机':1, '屏幕':2, '平板':3, '电视机':4,'门锁':5,'电脑':6,'其他':7}
        else:
            self.mappings = {'面膜/眼膜/唇膜':1, '蜜粉/散粉':2, '气垫BB/BB霜':3, '面部护肤':4,'粉扑':5,'身体护肤':6,'眉笔/眉粉/眉膏':7,'隔离霜/妆前乳':8,'粉底液/膏':9,'眼线笔':10,'腮红':11,'眼影':12,'手部护肤':13,'化妆刷':14,'粉饼':15,'卸妆液/油/水':16,'修容膏/笔':17,'遮瑕膏/笔':18,'其他':19}

        self.data = None
        self.train = None
        self.test = None
        self.flag = flag
        self.method = method
        self.is_val = is_val
        self.scene = scene
        if self.method == 'xgboost':
            self.params = { 'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.5],
                            'max_depth': [3, 5, 7, 15],
                            'n_estimators': [50, 100, 200, 300]
                             }
        elif self.method == 'lightgbm':
            self.params = { 'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.5],
                            'max_depth':  [3, 5, 7, 15],
                           'num_leaves': [30, 50, 100, 200]
                           }
        self.best_params = {}
        self.score = [np.inf, np.inf, np.inf, np.inf]
        self.model = None
        self.best_model = None
        self.test_size_dict = {'spu_day':30, 'spu_week':30, 'cls_day':30, 'cls_week':30}
        self.feat_size_dict = {'spu_day': 32, 'spu_week': 38, 'cls_day': 35, 'cls_week': 41}
        if self.feat == 'basic_feats':
            self.num_feat_dict = {'spu_day':['year','month','day','item_price',
                                        'past_1_d_sale_vol_sum','past_3_d_sale_vol_sum','past_7_d_sale_vol_sum','past_30_d_sale_vol_sum',
                                        'past_3_d_sale_vol_mean','past_7_d_sale_vol_mean','past_30_d_sale_vol_mean',
                                        'past_3_d_sale_vol_max','past_7_d_sale_vol_max','past_30_d_sale_vol_max',
                                        'past_3_d_sale_vol_min','past_7_d_sale_vol_min','past_30_d_sale_vol_min','past_7_d_sale_vol_std','past_30_d_sale_vol_std'],

                              'spu_week':['year','month','day','item_price',
                                        'past_1_w_sale_vol_sum','past_2_w_sale_vol_sum', 'past_3_w_sale_vol_sum','past_4_w_sale_vol_sum',
                                        'past_1_w_sale_vol_mean', 'past_2_w_sale_vol_mean','past_3_w_sale_vol_mean', 'past_4_w_sale_vol_mean',
                                        'past_1_w_sale_vol_max', 'past_2_w_sale_vol_max','past_3_w_sale_vol_max', 'past_4_w_sale_vol_max',
                                        'past_1_w_sale_vol_min', 'past_2_w_sale_vol_min','past_3_w_sale_vol_min', 'past_4_w_sale_vol_min',
                                        'past_1_w_sale_vol_std', 'past_2_w_sale_vol_std','past_3_w_sale_vol_std', 'past_4_w_sale_vol_std'],

                              'cls_day':['year','month','day','item_price_mean','item_price_max','item_price_min','item_spu_num',
                                         'past_1_d_sale_vol_sum', 'past_3_d_sale_vol_sum', 'past_7_d_sale_vol_sum','past_30_d_sale_vol_sum',
                                         'past_3_d_sale_vol_mean', 'past_7_d_sale_vol_mean', 'past_30_d_sale_vol_mean',
                                         'past_3_d_sale_vol_max', 'past_7_d_sale_vol_max', 'past_30_d_sale_vol_max',
                                         'past_3_d_sale_vol_min', 'past_7_d_sale_vol_min', 'past_30_d_sale_vol_min','past_7_d_sale_vol_std','past_30_d_sale_vol_std'],

                              'cls_week':['year','month','day','item_price_mean','item_price_max','item_price_min','item_spu_num',
                                          'past_1_w_sale_vol_sum', 'past_2_w_sale_vol_sum', 'past_3_w_sale_vol_sum','past_4_w_sale_vol_sum',
                                          'past_1_w_sale_vol_mean', 'past_2_w_sale_vol_mean', 'past_3_w_sale_vol_mean','past_4_w_sale_vol_mean',
                                          'past_1_w_sale_vol_max', 'past_2_w_sale_vol_max', 'past_3_w_sale_vol_max','past_4_w_sale_vol_max',
                                          'past_1_w_sale_vol_min', 'past_2_w_sale_vol_min', 'past_3_w_sale_vol_min','past_4_w_sale_vol_min',
                                          'past_1_w_sale_vol_std', 'past_2_w_sale_vol_std', 'past_3_w_sale_vol_std','past_4_w_sale_vol_std']
                              }
        elif self.feat == 'additional_feats':
            self.num_feat_dict = {'spu_day': ['year', 'month', 'day', 'item_price',
                                              'past_1_d_sale_vol_sum', 'past_3_d_sale_vol_sum', 'past_7_d_sale_vol_sum',
                                              'past_30_d_sale_vol_sum',
                                              'past_3_d_sale_vol_mean', 'past_7_d_sale_vol_mean',
                                              'past_30_d_sale_vol_mean',
                                              'past_3_d_sale_vol_max', 'past_7_d_sale_vol_max',
                                              'past_30_d_sale_vol_max',
                                              'past_3_d_sale_vol_min', 'past_7_d_sale_vol_min',
                                              'past_30_d_sale_vol_min', 'past_7_d_sale_vol_std',
                                              'past_30_d_sale_vol_std','item_weight_volume']}
        elif self.feat == 'optimized_feats':
            self.num_feat_dict = {'spu_day': ['month', 'day', 'item_price',
                                              'past_1_d_sale_vol_sum', 'past_3_d_sale_vol_sum', 'past_7_d_sale_vol_sum',
                                              'past_30_d_sale_vol_sum',
                                              'past_3_d_sale_vol_mean', 'past_7_d_sale_vol_mean',
                                              'past_30_d_sale_vol_mean',
                                              'past_3_d_sale_vol_max', 'past_7_d_sale_vol_max',
                                              'past_30_d_sale_vol_max',
                                              'past_3_d_sale_vol_min', 'past_7_d_sale_vol_min',
                                              'past_30_d_sale_vol_min', 'past_7_d_sale_vol_std',
                                              'past_30_d_sale_vol_std','item_weight_volume']}
        if self.feat == 'basic_feats':
            self.cls_feat_dict = {'spu_day':['is_workday','is_festival','festival_class','is_school_leave','school_leave_class','season_class',
                                         'is_pro_exact','pro_exact_class','is_pro_range','pro_range_class','item_class',
                                         'past_7_d_pay_time_max', 'past_30_d_pay_time_max'],

                              'spu_week':['has_festival', 'festival_class', 'has_school_leave', 'school_leave_class','season_class',
                                          'has_pro_exact', 'pro_exact_class', 'has_pro_range','pro_range_class', 'item_class',
                                          'past_1_w_pay_time_max', 'past_2_w_pay_time_max','past_3_w_pay_time_max', 'past_4_w_pay_time_max'],

                              'cls_day':['is_workday','is_festival','festival_class','is_school_leave','school_leave_class','season_class',
                                         'is_pro_exact','pro_exact_class','is_pro_range','pro_range_class', 'item_class',
                                         'past_7_d_pay_time_max','past_30_d_pay_time_max'],

                              'cls_week':['has_festival', 'festival_class', 'has_school_leave', 'school_leave_class','season_class',
                                          'has_pro_exact', 'pro_exact_class', 'has_pro_range','pro_range_class', 'item_class',
                                          'past_1_w_pay_time_max', 'past_2_w_pay_time_max', 'past_3_w_pay_time_max','past_4_w_pay_time_max']
                              }
        elif self.feat == 'additional_feats':
            self.cls_feat_dict = {
                'spu_day': ['is_workday', 'is_festival', 'festival_class', 'is_school_leave', 'school_leave_class',
                            'season_class',
                            'is_pro_exact', 'pro_exact_class', 'is_pro_range', 'pro_range_class', 'item_class',
                            'past_7_d_pay_time_max', 'past_30_d_pay_time_max','is_marketing','marketing_level','is_pro_single_item']
                }
        elif self.feat == 'optimized_feats':
            self.cls_feat_dict = {
                'spu_day': ['is_workday',
                            'is_pro_exact', 'item_class',
                            'marketing_level','is_pro_single_item']
                }
    def load_data(self):
        self.data = pd.read_excel(datapath)
        self.data.drop('index', axis=1, inplace=True)

    def data_size(self):
        if self.data is None:
            self.load_data()
        return self.data.shape[0] if self.data is not None else 0

    def split_data(self):

        if self.data is None:
            self.load_data()

        self.data['time'] = pd.to_datetime(self.data['year'].apply(str).str.cat(self.data['month'].apply(str), sep='-').str.cat(
            self.data['day'].apply(str), sep='-'), format='%Y-%m-%d')

        # data preprocessing
        # 数据标准化（不包括待预测列和分类类型列）
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(self.data[self.num_feat_dict[self.flag]])
        data_scaled = pd.DataFrame(data_scaled, columns=self.data[self.num_feat_dict[self.flag]].columns)
        if self.flag in ['spu_day','spu_week']:
            data_scaled = pd.concat([data_scaled, self.data[self.cls_feat_dict[self.flag]], self.data['sales'],self.data['time'], self.data['item']], axis=1).reset_index()
        else:
            data_scaled = pd.concat([data_scaled, self.data[self.cls_feat_dict[self.flag]], self.data['sales'],self.data['time']], axis=1).reset_index()
        # 对待预测列取对数处理
        data_scaled['sales'] = np.log1p(data_scaled['sales'])

        self.data = data_scaled

        # train和test数据集划分（按照样本的日期来划分，不是直接按照样本数量划分）
        self.data.sort_values(by=['time'],ascending=True,inplace=True)
        self.data = self.data.reset_index()
        self.data = self.data.drop(['index','level_0'], axis=1)

        self.test = self.data.loc[
            (((self.data.loc[self.data.index[-1],'time'] - self.data['time']).apply(lambda x: x.days) <= self.test_size_dict[self.flag]) &
             ((self.data.loc[self.data.index[-1],'time'] - self.data['time']).apply(lambda x: x.days) >= 0))]

        self.train = self.data.loc[(self.data.loc[self.data.index[-1], 'time'] - self.data['time']).apply(lambda x: x.days) > self.test_size_dict[self.flag]]

        # shuffle
        self.train = shuffle(self.train)

    def run(self, grid_search=False, retrain=False):
        if self.train is None or self.test is None:
            self.split_data()

        if retrain or self.model is None:
            if self.method == 'xgboost':
                self.model = XGBRegressor(learning_rate=0.1,
                                   n_estimators=150,
                                   max_depth=5,
                                   min_child_weight=1,
                                   gamma=0,
                                   subsample=0.8,
                                   colsample_bytree=0.8,
                                   objective='reg:squarederror',
                                   reg_alpha=0,
                                   reg_lambda=1,
                                   nthread=4,
                                   scale_pos_weight=1,
                                   seed=27)

            elif self.method == 'lightgbm':
                self.model = LGBMRegressor(
                    boosting_type='gbdt',
                    num_leaves=31,
                    max_depth=-1,
                    learning_rate=0.1,
                    n_estimators=100,
                    objective='regression',
                    min_split_gain=0.0,
                    min_child_samples=20,
                    subsample=1.0,
                    subsample_freq=0,
                    colsample_bytree=1.0,
                    reg_alpha=0.0,
                    reg_lambda=0.0,
                    random_state=None,
                    silent=True)

        if not grid_search:
            if self.flag in ['spu_day', 'spu_week']:
                self.model.fit(self.train.drop(['sales','time','item'], axis=1), self.train['sales'])
            else:
                self.model.fit(self.train.drop(['sales', 'time'], axis=1), self.train['sales'])

    def metrics(self, data, is_all_sample):
        n = len(data)  # 样本数量
        print('测试样本数量: %d' % n)
        if self.flag in ['spu_day', 'spu_week']:
            y_pred = self.model.predict(data.drop(['sales','time','item'], axis=1))
        else:
            y_pred = self.model.predict(data.drop(['sales', 'time'], axis=1))
        mae = mean_absolute_error(data['sales'], y_pred)
        print('MAE: %.4f' % mae)
        mape = mean_absolute_percentage_error(data['sales'], y_pred)
        print('MAPE: %.4f' % mape)
        rmse = sqrt(mean_squared_error(data['sales'], y_pred))
        print('RMSE: %.4f' % rmse)

        R2 = r2_score(data['sales'], y_pred)
        print('R2: %.4f' % R2)

        feat_p = self.feat_size_dict[self.flag] # 特征维度
        if (n - feat_p - 1) == 0:
            adjusted_R2 = np.nan
        else:
            adjusted_R2 = 1 - ((1 - R2) * (n - 1)) / (n - feat_p - 1) # 校正决定系数R2，抵消了样本数量对R2的影响
        print('adjusted_R2: %.4f' % adjusted_R2)

        real_sales = np.expm1(data['sales']).sum()
        pred_sales = np.expm1(pd.DataFrame(y_pred, columns=['prediction'])).sum()
        AE = real_sales/pred_sales
        print('实际销量: %.4f' % real_sales)
        print('预测销量: %.4f' % pred_sales)
        print('AE ratio: %.4f' % AE['prediction'])

        if is_all_sample:
            data.reset_index(drop=True, inplace=True)
            output = pd.concat([data, pd.DataFrame(y_pred, columns=['prediction'])], axis=1)
            output['prediction'] = np.expm1(output['prediction'])
            output = output.groupby(['time']).agg({'prediction': {'sum'}})
            writer = pd.ExcelWriter(output_dir + '\\' "predictions.xlsx")
            output.to_excel(writer)
            writer._save()
            writer.close()

        return mae, mape, rmse, R2, adjusted_R2, AE['prediction']


    def predict(self):
        if self.train is None:
            self.split_data()
        if self.model is None:
            self.run()

        metrics_dict = {'MAE':[],'MAPE':[],'RMSE':[],'R2':[],'adjusted_R2':[],'AE':[]}
        data_dict = {'train':[], 'test':[]}
        names = []

        print('训练样本数量: %d' % len(self.train))
        print('测试集的整体评估结果如下:')
        metrics_all = self.metrics(self.test, True)
        names.append('所有样本')
        data_dict['train'].append(len(self.train))
        data_dict['test'].append(len(self.test))
        metrics_dict['MAE'].append(round(metrics_all[0],4))
        metrics_dict['MAPE'].append(round(metrics_all[1],4))
        metrics_dict['RMSE'].append(round(metrics_all[2],4))
        metrics_dict['R2'].append(round(metrics_all[3],4))
        metrics_dict['adjusted_R2'].append(round(metrics_all[4],4))
        metrics_dict['AE'].append(round(metrics_all[5],4))

        # 各个类别的指标
        print('测试集中各个大类的评估结果如下:')
        item_cls = list(self.test.loc[:, 'item_class'].unique())
        mappings_reverse = {value: key for key, value in self.mappings.items()}
        for c in item_cls:
            cls_qp = self.test.loc[(self.test['item_class'] == c)]
            print('class: %s' % str(mappings_reverse[c]))
            print('训练样本数量: %d' % len(self.train.loc[(self.train['item_class'] == c)]))
            metrics_cls = self.metrics(cls_qp, False)
            names.append(str(mappings_reverse[c]))
            data_dict['train'].append(len(self.train.loc[(self.train['item_class'] == c)]))
            data_dict['test'].append(len(cls_qp))
            metrics_dict['MAE'].append(round(metrics_cls[0],4))
            metrics_dict['MAPE'].append(round(metrics_cls[1],4))
            metrics_dict['RMSE'].append(round(metrics_cls[2],4))
            metrics_dict['R2'].append(round(metrics_cls[3],4))
            metrics_dict['adjusted_R2'].append(round(metrics_cls[4],4))
            metrics_dict['AE'].append(round(metrics_cls[5],4))

        # 过去一段时间内的销量top 10的sku或spu的指标
        if self.flag in ['spu_day', 'spu_week']:
            temp = pd.DataFrame(self.data.groupby('item').agg({'sales': {'sum'}})).reset_index()
            temp.columns = [i[0] + "_" + i[1] for i in temp.columns]
            temp.sort_values(by=['sales_sum'], ascending=False, inplace=True)
            item_spu = list(temp.head(30)['item_'].unique())
            n = 0
            for c in item_spu:
                spu_qp = self.test.loc[(self.test['item'] == c)]
                if len(spu_qp) == 0:
                    continue
                if n < 10:
                    print('spu: %s' % c)
                    print('训练样本数量: %d' % len(self.train.loc[(self.train['item'] == c)]))
                    metrics_spu = self.metrics(spu_qp, False)
                    names.append(c)
                    data_dict['train'].append(len(self.train.loc[(self.train['item'] == c)]))
                    data_dict['test'].append(len(spu_qp))
                    metrics_dict['MAE'].append(round(metrics_spu[0],4))
                    metrics_dict['MAPE'].append(round(metrics_spu[1],4))
                    metrics_dict['RMSE'].append(round(metrics_spu[2],4))
                    metrics_dict['R2'].append(round(metrics_spu[3],4))
                    metrics_dict['adjusted_R2'].append(round(metrics_spu[4],4))
                    metrics_dict['AE'].append(round(metrics_spu[5],4))
                    n = n + 1

        # plot
        # metrics bar
        for metric in metrics_dict.keys():

            plt.clf()
            df = pd.DataFrame({'data': metrics_dict[metric]}, index=names)
            if self.scene == 'jiadian':
                ax = df.plot(kind='bar', color='red',linewidth=5,figsize=(10,3))
            else:
                ax = df.plot(kind='bar', color='red', linewidth=5, figsize=(15, 3))
            for i, val in enumerate(df['data'].values):
                ax.text(i, val, val, horizontalalignment='center', verticalalignment='bottom')
            ax.set(title=metric + '评估指标结果', ylabel=metric)
            plt.tick_params(axis='x', labelrotation=90)
            plt.savefig(output_dir + '\\' + metric + '评估指标结果.png', bbox_inches='tight')
            plt.show()

        # sales vs predictions (class)
        if self.flag in ['spu_day', 'spu_week']:
            predictions = self.model.predict(self.test.drop(['sales','time','item'], axis=1))
        else:
            predictions = self.model.predict(self.test.drop(['sales','time'], axis=1))

        mappings_reverse = {value: key for key, value in self.mappings.items()}
        self.test['item_class'].replace(mappings_reverse, inplace=True)
        self.test.reset_index(drop=True, inplace=True)
        self.test = pd.concat([self.test, pd.DataFrame(predictions, columns=['prediction'])], axis=1)

        self.test['prediction'] = np.expm1(self.test['prediction'])
        self.test['sales'] = np.expm1(self.test['sales'])
        plt.clf()
        plt.figure()
        if self.scene == 'jiadian':
            draw_set = {'耳机': ['gray','s'], '屏幕': ['green','*'], '平板': ['blue','o'], '电视机': ['olive','^'], '门锁': ['purple','p'], '电脑': ['pink','X'], '其他': ['brown','+']}
        else:
            draw_set = {'面膜/眼膜/唇膜': ['gray','s'], '蜜粉/散粉': ['green','*'], '气垫BB/BB霜': ['blue','o'], '面部护肤': ['olive','^'], '粉扑': ['purple','p'], '身体护肤': ['pink','X'], '眉笔/眉粉/眉膏': ['brown','+'], '隔离霜/妆前乳': ['rosybrown','v'], '粉底液/膏': ['lime','<'], '眼线笔': ['coral','>'], '腮红': ['gold','1'], '眼影': ['seagreen','2'], '手部护肤': ['cyan','3'], '化妆刷': ['cadetblue','4'], '粉饼': ['cornflowerblue','P'], '卸妆液/油/水': ['navy','h'], '修容膏/笔': ['indigo','x'], '遮瑕膏/笔': ['plum','d'], '其他': ['maroon','D']}

        item_cls = list(self.test.loc[:, 'item_class'].unique())
        for c in item_cls:
            plt.scatter(self.test.loc[(self.test['item_class'] == c),'sales'], self.test.loc[(self.test['item_class'] == c),'prediction'], c=draw_set[c][0], label=c, marker=draw_set[c][1])

        plt.legend(loc='best')
        plt.title(self.flag+ "数据的" +self.method + "模型预测结果")
        plt.xlabel('真实销量')
        plt.ylabel('预测销量')
        # 定义 y=x 线的 x 和 y 值
        x_line = np.linspace(0, 500, 50000)
        y_line = x_line
        # 绘制 y=x 线
        plt.plot(x_line, y_line, '--')

        plt.savefig(output_dir + '\\' + '模型预测结果.png', bbox_inches='tight')
        plt.show()

        temp_list = [names] + [data_dict['train']] + [data_dict['test']]+[metrics_dict['MAE']]+ [metrics_dict['MAPE']]+ [metrics_dict['RMSE']]+ [metrics_dict['R2']]+ [metrics_dict['adjusted_R2']]+ [metrics_dict['AE']]
        temp_list = list(map(list, zip(*temp_list)))  # 转置
        output = pd.DataFrame(temp_list,columns=['name','train','test','MAE','MAPE','RMSE','R2','adjusted_R2','AE'])
        writer = pd.ExcelWriter(output_dir + '\\' "metrics.xlsx")
        output.to_excel(writer)
        writer._save()
        writer.close()

    def grid_search(self):
        print('begin grid search...')
        self.run(True, True)
        grid_search = None
        grid_search = GridSearchCV(estimator=self.model,
                               param_grid=self.params,
                                scoring='r2',
                               #n_jobs=-1,
                               cv=5)
        if self.flag in ['spu_day', 'spu_week']:
            grid_search.fit(self.train.drop(['sales','time','item'], axis=1), self.train['sales'])
        else:
            grid_search.fit(self.train.drop(['sales','time'], axis=1), self.train['sales'])

        print('finish grid search, result as follows:')
        # 输出最佳参数
        self.best_params = grid_search.best_params_
        print("Best Parameters:", self.best_params)
        self.best_model = grid_search.best_estimator_
        print('Best r2:', grid_search.best_score_)


    def process(self):

        if not is_val:
            print('------- best result from grid search and k-fold ------------')
            self.grid_search()
            self.model = self.best_model
            self.save_model(self.flag)
            self.predict()
        else:
            self.model = self.load_model(self.flag).model
            self.predict()

        print('analyze feature importance and contributions for model result:')
        # feature importance
        if self.method == 'xgboost':
            plot_importance_xgb(self.model, title='特征重要性排序', xlabel='得分', ylabel='特征', grid=False)
        elif self.method == 'lightgbm':
            plot_importance_gbm(self.model, title='特征重要性排序', xlabel='得分', ylabel='特征', grid=False)
        plt.savefig(output_dir + '\\' + '特征重要性.png', bbox_inches='tight')
        plt.show()

        # # feature contributions based on SHAP value
        # # 创建SHAP解释器
        # explainer = shap.TreeExplainer(self.model)
        # # 计算SHAP值
        # mappings = {'耳机':1, '屏幕':2, '平板':3, '电视机':4,'门锁':5,'电脑':6,'其他':7}
        # self.test['item_class'].replace(mappings, inplace=True)
        # shap_values = explainer.shap_values(self.test.drop(['sales','prediction'], axis=1))
        # # 可视化SHAP值 # TODO: bug待解决：shap.summary_plot报错，“AssertionError: The shape of the shap_values matrix does not match the shape of the provided data matrix.”
        # shap.summary_plot(shap_values, self.test)


    def save_model(self, name):
        with open(output_dir + '\\' + f'{name}.pkl', 'wb') as fp:
            pickle.dump(self, fp)

    def load_model(self, name):
        with open(output_dir + '\\' + f'{name}.pkl', 'rb') as fp:
            return pickle.load(fp)


if __name__ == '__main__':

    flag = 'spu_day' # 特征粒度：spu_day, spu_week, cls_day, cls_week
    method = 'xgboost' # 机器学习模型xgboost：, lightgbm
    is_val = True
    scene = 'meizhuang' # meizhuang, jiadian
    feat = 'additional_feats' # ['basic_feats','additional_feats','optimized_feats']
    print(flag,method,scene,feat)

    ml = MLTrain(flag, method, is_val, scene, feat)
    ml.process()
    print('all finished!')


