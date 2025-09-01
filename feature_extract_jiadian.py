import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import matplotlib
from datetime import *
from utils.time_process import DateProcess
import numpy as np
matplotlib.use('TkAgg')
pd.options.display.precision = 4
from pandas.api.types import CategoricalDtype
from pandas.core.frame import DataFrame


flag = 'spu粒度_天'
item_cls = '耳机'

def classify(x):
    cls1 = ['耳机','FreeBuds']
    cls2 = ['智慧屏']
    cls3 = ['Pad','平板','MateBook']
    cls4 = ['电视机']
    cls5 = ['门锁']
    cls6 = ['笔记本','电脑']

    if any([keyword in x for keyword in cls1]):
        x = '耳机'
    elif any([keyword in x for keyword in cls2]):
        x = '屏幕'
    elif any([keyword in x for keyword in cls3]):
        x = '平板'
    elif any([keyword in x for keyword in cls4]):
        x = '电视机'
    elif any([keyword in x for keyword in cls5]):
        x = '门锁'
    elif any([keyword in x for keyword in cls6]):
        x = '电脑'
    else:
        x = '其他'
    return x

def pay_range(x):
    range = ''
    hour = pd.to_datetime(x).hour
    if hour >= 6 and hour <= 12:
        range = 'morning'
    elif hour > 12 and hour <= 18:
        range = 'afternoon'
    elif hour > 18 and hour < 24:
        range = 'night'
    elif hour < 6:
        range = 'wee_hours'

    return range

def date_2_week(group):
    has_festival = 0
    festival_class = 0
    has_school_leave = 0
    school_leave_class = 0
    season_class = 1
    has_pro_exact = 0
    pro_exact_class = 0
    has_pro_range = 0
    pro_range_class = 0
    for i in group.index:
        if group.loc[i, 'is_festival'] == 1:
            has_festival = 1
        if group.loc[i, 'festival_class'] != 0:
            festival_class = group.loc[i, 'festival_class']
        if group.loc[i, 'is_school_leave'] == 1:
            has_school_leave = 1
        if group.loc[i, 'school_leave_class'] != 0:
            school_leave_class = group.loc[i, 'school_leave_class']
        season_class = group.loc[i, 'season_class']
        if group.loc[i, 'is_pro_exact'] == 1:
            has_pro_exact = 1
        if group.loc[i, 'pro_exact_class'] != 0:
            pro_exact_class = group.loc[i, 'pro_exact_class']
        if group.loc[i, 'is_pro_range'] == 1:
            has_pro_range = 1
        if group.loc[i, 'pro_range_class'] != 0:
            pro_range_class = group.loc[i, 'pro_range_class']

    output = {'has_festival':[has_festival],'festival_class':[festival_class], 'has_school_leave':[has_school_leave],
              'school_leave_class':[school_leave_class], 'season_class':[season_class], 'has_pro_exact':[has_pro_exact],
              'pro_exact_class':[pro_exact_class], 'has_pro_range':[has_pro_range], 'pro_range_class':[pro_range_class]}

    return pd.DataFrame.from_dict(output)

def feature_process(df, flag):

    df['date'] = pd.to_datetime(df['付款时间']).dt.day.apply(str).str.cat(
        pd.to_datetime(df['付款时间']).dt.month.apply(str), sep=',').str.cat(
        pd.to_datetime(df['付款时间']).dt.year.apply(str), sep=',')

    df['week'] = pd.to_datetime(df['付款时间']).dt.isocalendar().week.apply(str).str.cat(
        pd.to_datetime(df['付款时间']).dt.year.apply(str), sep=',')

    # 付款时段
    df['付款时段'] = df['付款时间'].apply(pay_range)
    features = None

    if flag == '0':
        temp1 = pd.DataFrame(df[['date','商品名称','付款时段']].groupby(['date','商品名称']).agg(lambda x: x.value_counts().index[0]))
        temp = pd.DataFrame(df.groupby(['date','商品名称']).agg({'数量': {'sum'}, '付款时间': {'min'}, '实际单价': {'min'}}))
        temp = pd.concat([temp,temp1],axis=1).reset_index()

        # 重命名列名
        temp.columns = [i[0] + "_" + i[1] for i in temp.columns]
        temp.sort_values(by=['付款时间_min'], ascending=True, inplace=True)
        temp['year'] = pd.to_datetime(temp['付款时间_min']).dt.year
        temp['month'] = pd.to_datetime(temp['付款时间_min']).dt.month
        temp['day'] = pd.to_datetime(temp['付款时间_min']).dt.day

        DP = DateProcess(temp['付款时间_min'])
        df_day_label = DP.get_Feature_day()
        df_day_label['is_workday'] = df_day_label.apply(lambda x: 1 if x.day_type_chinese == '工作日' else 0, axis=1)

        # is_one_hot（暂不考虑）
        # cat_type = CategoricalDtype(categories=['非假期','元旦','春节','清明节','劳动节','端午节','中秋节','国庆节','情人节','妇女节','母亲节','七夕节'], ordered=True)
        # df_day_label['temp'] = df_day_label.apply(lambda x: '非假期' if x.day_type_chinese == '工作日' or x.day_type_chinese == '周末或调休' else x.day_type_chinese, axis=1).astype(cat_type)
        # dummy1 = pd.get_dummies(df_day_label['temp'], dummy_na=False, drop_first=False)
        # dummy1.rename(columns={'非假期': 'not_fest','元旦': 'yuandan', '春节': 'chunjie','清明节': 'qingming', '劳动节': 'laodong','端午节': 'duanwu', '中秋节': 'zhongqiu','国庆节': 'guoqing', '情人节': 'qingren', '妇女节': 'funv','母亲节': 'muqin', '七夕节': 'qixi'}, inplace=True)
        # for u in dummy1.columns:
        #     if dummy1[u].dtype == bool:
        #         dummy1[u] = dummy1[u].astype('int')

        df_day_label['is_festival'] = df_day_label.apply(lambda x: 0 if x.day_type_chinese == '工作日' or x.day_type_chinese == '周末或调休' else 1,axis=1)
        mappings = {'工作日': 0, '周末或调休': 0, '元旦': 1, '春节': 2, '清明节': 3, '劳动节': 4,'端午节': 5, '中秋节': 6, '国庆节': 7, '情人节': 8, '妇女节': 6, '母亲节': 7, '七夕节': 8}
        df_day_label['day_type_chinese'].replace(mappings, inplace=True)
        df_day_label.rename(columns={'day_type_chinese': 'festival_class'}, inplace=True)

        df_day_label['is_school_leave'] = df_day_label.apply(lambda x: 0 if x.school_leave == '' else 1,axis=1)
        df_day_label['school_leave'] = df_day_label.apply(lambda x: '非寒暑假' if x.school_leave == '' else x.school_leave,axis=1)
        mappings = {'非寒暑假': 0, '寒假': 1, '暑假': 2}
        df_day_label['school_leave'].replace(mappings, inplace=True)
        df_day_label.rename(columns={'school_leave': 'school_leave_class'}, inplace=True)

        mappings = {'1': 1, '2': 2, '3': 3, '4':4}
        df_day_label['quarter'].replace(mappings, inplace=True)
        df_day_label.rename(columns={'quarter': 'season_class'}, inplace=True)

        df_day_label['is_pro_exact'] = df_day_label.apply(lambda x: 0 if x.sales_type_exact == '' else 1,axis=1)
        df_day_label['sales_type_exact'] = df_day_label.apply(lambda x: '非精确促销' if x.sales_type_exact == '' else x.sales_type_exact,axis=1)
        mappings = {'非精确促销':0, '双11':1, '双12':2, '618':3}
        df_day_label['sales_type_exact'].replace(mappings, inplace=True)
        df_day_label.rename(columns={'sales_type_exact': 'pro_exact_class'}, inplace=True)

        df_day_label['is_pro_range'] = df_day_label.apply(lambda x: 0 if x.sales_type_range == '' else 1,axis=1)
        df_day_label['sales_type_range'] = df_day_label.apply(lambda x: '非范围促销' if x.sales_type_range == '' else x.sales_type_range,axis=1)
        mappings = {'非范围促销':0, '双11':1, '双12':2, '618':3}
        df_day_label['sales_type_range'].replace(mappings, inplace=True)
        df_day_label.rename(columns={'sales_type_range': 'pro_range_class'}, inplace=True)

        temp['商品类别'] = temp['商_品'].apply(classify)
        mappings = {'耳机':1, '屏幕':2, '平板':3, '电视机':4,'门锁':5,'电脑':6,'其他':7}
        temp['商品类别'].replace(mappings, inplace=True)
        temp.rename(columns={'商品类别': 'item_class'}, inplace=True)

        mappings = {'morning':1, 'afternoon':2, 'night':3, 'wee_hours':4}
        temp['付_款'].replace(mappings, inplace=True)

        temp['time'] = pd.to_datetime(temp['付款时间_min'], format='%Y-%m-%d')
        temp = pd.concat([temp,df_day_label[['is_workday','is_festival','festival_class','is_school_leave','school_leave_class','season_class','is_pro_exact','pro_exact_class','is_pro_range','pro_range_class']]],axis=1)

        # 历史累计
        output = pd.DataFrame(columns=temp.columns.tolist())

        sku = list(temp.loc[:,'商_品'].unique())
        for c in sku:
            sku_qp = temp.loc[(temp['商_品'] == c)].reset_index()
            for i in sku_qp.index:
                # 对近1天数据进行切片
                day1_qiepian = sku_qp.loc[
                    (((sku_qp.loc[i,'time']-sku_qp['time']).apply(lambda x: x.days) <= 1) &
                     ((sku_qp.loc[i,'time']-sku_qp['time']).apply(lambda x: x.days) > 0))]
                day1_sum = day1_qiepian['数量_sum'].sum()
                sku_qp.loc[i,'past_1_d_sale_vol_sum']=day1_sum

                # 对近3天数据进行切片
                day3_qiepian = sku_qp.loc[
                    (((sku_qp.loc[i,'time']-sku_qp['time']).apply(lambda x: x.days) <= 3) &
                     ((sku_qp.loc[i,'time']-sku_qp['time']).apply(lambda x: x.days) > 0))]
                day3_sum = day3_qiepian['数量_sum'].sum()
                sku_qp.loc[i,'past_3_d_sale_vol_sum']=day3_sum
                day3_mean = day3_qiepian['数量_sum'].mean()
                sku_qp.loc[i, 'past_3_d_sale_vol_mean'] = day3_mean
                day3_max = day3_qiepian['数量_sum'].max()
                sku_qp.loc[i, 'past_3_d_sale_vol_max'] = day3_max
                day3_min = day3_qiepian['数量_sum'].min()
                sku_qp.loc[i, 'past_3_d_sale_vol_min'] = day3_min
                day3_std = np.array(day3_qiepian['数量_sum']).std()
                sku_qp.loc[i, 'past_3_d_sale_vol_std'] = day3_std
                # 对7天数据进行切片
                day7_qiepian = sku_qp.loc[
                    (((sku_qp.loc[i,'time']-sku_qp['time']).apply(lambda x: x.days) <= 7) &
                     ((sku_qp.loc[i,'time']-sku_qp['time']).apply(lambda x: x.days) > 0))]
                day7_sum = day7_qiepian['数量_sum'].sum()
                sku_qp.loc[i, 'past_7_d_sale_vol_sum'] = day7_sum
                day7_mean = day7_qiepian['数量_sum'].mean()
                sku_qp.loc[i, 'past_7_d_sale_vol_mean'] = day7_mean
                day7_max = day7_qiepian['数量_sum'].max()
                sku_qp.loc[i, 'past_7_d_sale_vol_max'] = day7_max
                day7_min = day7_qiepian['数量_sum'].min()
                sku_qp.loc[i, 'past_7_d_sale_vol_min'] = day7_min
                day7_std = np.array(day7_qiepian['数量_sum']).std()
                sku_qp.loc[i, 'past_7_d_sale_vol_std'] = day7_std
                if len(day7_qiepian) != 0:
                    day7_pay = day7_qiepian['付_款'].value_counts().index[0]
                else:
                    day7_pay = np.nan
                sku_qp.loc[i, 'past_7_d_pay_time_max'] = day7_pay

                # 对30天数据进行切片
                day30_qiepian = sku_qp.loc[
                    (((sku_qp.loc[i,'time']-sku_qp['time']).apply(lambda x: x.days) <= 30) &
                     ((sku_qp.loc[i,'time']-sku_qp['time']).apply(lambda x: x.days) > 0))]
                day30_sum = day30_qiepian['数量_sum'].sum()
                sku_qp.loc[i, 'past_30_d_sale_vol_sum'] = day30_sum
                day30_mean = day30_qiepian['数量_sum'].mean()
                sku_qp.loc[i, 'past_30_d_sale_vol_mean'] = day30_mean
                day30_max = day30_qiepian['数量_sum'].max()
                sku_qp.loc[i, 'past_30_d_sale_vol_max'] = day30_max
                day30_min = day30_qiepian['数量_sum'].min()
                sku_qp.loc[i, 'past_30_d_sale_vol_min'] = day30_min
                day30_std = np.array(day30_qiepian['数量_sum']).std()
                sku_qp.loc[i, 'past_30_d_sale_vol_std'] = day30_std
                if len(day30_qiepian) != 0:
                    day30_pay = day30_qiepian['付_款'].value_counts().index[0]
                else:
                    day30_pay = np.nan
                sku_qp.loc[i, 'past_30_d_pay_time_max'] = day30_pay

            output = output._append(sku_qp, ignore_index=True)

        output.rename(columns={'商_品': 'item','实际单价_min':'item_price','数量_sum':'sales'}, inplace=True)
        output = output.fillna(0)

        # 因为有前30天的累计特征，所以需要把数据集最前面30天的数据都扔掉
        output.sort_values(by=['time'], ascending=True, inplace=True)
        output = output.iloc[3:] # 前三行数据日期不连续，直接扔掉
        output['diff'] = output['time'].sub(output['time'].iloc[0]).apply(lambda x: x.days)
        output['is_pre_30d'] = output['diff'] <= 30
        output = output[output['is_pre_30d'] == False]

        features = output[['sales','year','month','day','is_workday',
                    'is_festival','festival_class','is_school_leave','school_leave_class','season_class','is_pro_exact','pro_exact_class','is_pro_range','pro_range_class','item','item_class','item_price',
                    'past_1_d_sale_vol_sum','past_3_d_sale_vol_sum','past_7_d_sale_vol_sum','past_30_d_sale_vol_sum',
                                             'past_3_d_sale_vol_mean','past_7_d_sale_vol_mean','past_30_d_sale_vol_mean',
                                             'past_3_d_sale_vol_max','past_7_d_sale_vol_max','past_30_d_sale_vol_max',
                                             'past_3_d_sale_vol_min','past_7_d_sale_vol_min','past_30_d_sale_vol_min','past_7_d_sale_vol_std',
                                             'past_30_d_sale_vol_std','past_7_d_pay_time_max','past_30_d_pay_time_max']]

    elif flag == '1':
        temp1 = pd.DataFrame(
            df[['week', '商品名称', '付款时段']].groupby(['week', '商品名称']).agg(lambda x: x.value_counts().index[0]))
        temp2 = pd.DataFrame(
            df.groupby(['week', '商品名称']).agg({'数量': {'sum'}, '付款时间': {'min'}, '实际单价': {'min'}}))
        temp3 = pd.concat([temp2, temp1], axis=1).reset_index()
        temp3.columns = [i[0] + "_" + i[1] for i in temp3.columns]
        temp3['year'] = pd.to_datetime(temp3['付款时间_min']).dt.year
        temp3['month'] = pd.to_datetime(temp3['付款时间_min']).dt.month
        temp3['day'] = pd.to_datetime(temp3['付款时间_min']).dt.day

        temp3['商品类别'] = temp3['商_品'].apply(classify)
        mappings = {'耳机': 1, '屏幕': 2, '平板': 3, '电视机': 4, '门锁': 5, '电脑': 6, '其他': 7}
        temp3['商品类别'].replace(mappings, inplace=True)
        temp3.rename(columns={'商品类别': 'item_class'}, inplace=True)

        mappings = {'morning': 1, 'afternoon': 2, 'night': 3, 'wee_hours': 4}
        temp3['付_款'].replace(mappings, inplace=True)

        temp = pd.DataFrame(
            df.groupby(['date', '商品名称']).agg({'付款时间': {'min'},'week': {'first'}}))
        temp.columns = [i[0] + "_" + i[1] for i in temp.columns]
        DP = DateProcess(temp['付款时间_min'])
        df_day_label = DP.get_Feature_day()
        df_day_label['is_workday'] = df_day_label.apply(lambda x: 1 if x.day_type_chinese == '工作日' else 0, axis=1)
        df_day_label['is_festival'] = df_day_label.apply(
            lambda x: 0 if x.day_type_chinese == '工作日' or x.day_type_chinese == '周末或调休' else 1, axis=1)
        mappings = {'工作日': 0, '周末或调休': 0, '元旦': 1, '春节': 2, '清明节': 3, '劳动节': 4, '端午节': 5,
                    '中秋节': 6, '国庆节': 7, '情人节': 8, '妇女节': 6, '母亲节': 7, '七夕节': 8}
        df_day_label['day_type_chinese'].replace(mappings, inplace=True)
        df_day_label.rename(columns={'day_type_chinese': 'festival_class'}, inplace=True)

        df_day_label['is_school_leave'] = df_day_label.apply(lambda x: 0 if x.school_leave == '' else 1, axis=1)
        df_day_label['school_leave'] = df_day_label.apply(
            lambda x: '非寒暑假' if x.school_leave == '' else x.school_leave, axis=1)
        mappings = {'非寒暑假': 0, '寒假': 1, '暑假': 2}
        df_day_label['school_leave'].replace(mappings, inplace=True)
        df_day_label.rename(columns={'school_leave': 'school_leave_class'}, inplace=True)

        mappings = {'1': 1, '2': 2, '3': 3, '4': 4}
        df_day_label['quarter'].replace(mappings, inplace=True)
        df_day_label.rename(columns={'quarter': 'season_class'}, inplace=True)

        df_day_label['is_pro_exact'] = df_day_label.apply(lambda x: 0 if x.sales_type_exact == '' else 1, axis=1)
        df_day_label['sales_type_exact'] = df_day_label.apply(
            lambda x: '非精确促销' if x.sales_type_exact == '' else x.sales_type_exact, axis=1)
        mappings = {'非精确促销': 0, '双11': 1, '双12': 2, '618': 3}
        df_day_label['sales_type_exact'].replace(mappings, inplace=True)
        df_day_label.rename(columns={'sales_type_exact': 'pro_exact_class'}, inplace=True)

        df_day_label['is_pro_range'] = df_day_label.apply(lambda x: 0 if x.sales_type_range == '' else 1, axis=1)
        df_day_label['sales_type_range'] = df_day_label.apply(
            lambda x: '非范围促销' if x.sales_type_range == '' else x.sales_type_range, axis=1)
        mappings = {'非范围促销': 0, '双11': 1, '双12': 2, '618': 3}
        df_day_label['sales_type_range'].replace(mappings, inplace=True)
        df_day_label.rename(columns={'sales_type_range': 'pro_range_class'}, inplace=True)

        temp = pd.concat([temp, df_day_label], axis=1)

        temp4 = temp.groupby(['week_first', '商品名称']).apply(date_2_week).reset_index()

        temp3['time'] = pd.to_datetime(temp3['付款时间_min'], format='%Y-%m-%d')
        temp3 = pd.concat([temp3, temp4], axis=1)

        # 历史累计
        output = pd.DataFrame(columns=temp3.columns.tolist())

        sku = list(temp3.loc[:, '商_品'].unique())
        for c in sku:
            sku_qp = temp3.loc[(temp3['商_品'] == c)].reset_index()
            for i in sku_qp.index:
                # 对近1周数据进行切片
                week1_qiepian = sku_qp.loc[
                    (((sku_qp.loc[i, 'time'] - sku_qp['time']).apply(lambda x: x.days) <= 7) &
                     ((sku_qp.loc[i, 'time'] - sku_qp['time']).apply(lambda x: x.days) > 0))]
                week1_sum = week1_qiepian['数量_sum'].sum()
                sku_qp.loc[i, 'past_1_w_sale_vol_sum'] = week1_sum
                week1_mean = week1_qiepian['数量_sum'].mean()
                sku_qp.loc[i, 'past_1_w_sale_vol_mean'] = week1_mean
                week1_max = week1_qiepian['数量_sum'].max()
                sku_qp.loc[i, 'past_1_w_sale_vol_max'] = week1_max
                week1_min = week1_qiepian['数量_sum'].min()
                sku_qp.loc[i, 'past_1_w_sale_vol_min'] = week1_min
                week1_std = np.array(week1_qiepian['数量_sum']).std()
                sku_qp.loc[i, 'past_1_w_sale_vol_std'] = week1_std
                if len(week1_qiepian) != 0:
                    week1_pay = week1_qiepian['付_款'].value_counts().index[0]
                else:
                    week1_pay = np.nan
                sku_qp.loc[i, 'past_1_w_pay_time_max'] = week1_pay

                # 对近2周数据进行切片
                week2_qiepian = sku_qp.loc[
                    (((sku_qp.loc[i, 'time'] - sku_qp['time']).apply(lambda x: x.days) <= 14) &
                     ((sku_qp.loc[i, 'time'] - sku_qp['time']).apply(lambda x: x.days) > 0))]
                week2_sum = week2_qiepian['数量_sum'].sum()
                sku_qp.loc[i, 'past_2_w_sale_vol_sum'] = week2_sum
                week2_mean = week2_qiepian['数量_sum'].mean()
                sku_qp.loc[i, 'past_2_w_sale_vol_mean'] = week2_mean
                week2_max = week2_qiepian['数量_sum'].max()
                sku_qp.loc[i, 'past_2_w_sale_vol_max'] = week2_max
                week2_min = week2_qiepian['数量_sum'].min()
                sku_qp.loc[i, 'past_2_w_sale_vol_min'] = week2_min
                week2_std = np.array(week2_qiepian['数量_sum']).std()
                sku_qp.loc[i, 'past_2_w_sale_vol_std'] = week2_std
                if len(week2_qiepian) != 0:
                    week2_pay = week2_qiepian['付_款'].value_counts().index[0]
                else:
                    week2_pay = np.nan
                sku_qp.loc[i, 'past_2_w_pay_time_max'] = week2_pay

                # 对近3周数据进行切片
                week3_qiepian = sku_qp.loc[
                    (((sku_qp.loc[i, 'time'] - sku_qp['time']).apply(lambda x: x.days) <= 21) &
                     ((sku_qp.loc[i, 'time'] - sku_qp['time']).apply(lambda x: x.days) > 0))]
                week3_sum = week3_qiepian['数量_sum'].sum()
                sku_qp.loc[i, 'past_3_w_sale_vol_sum'] = week3_sum
                week3_mean = week3_qiepian['数量_sum'].mean()
                sku_qp.loc[i, 'past_3_w_sale_vol_mean'] = week3_mean
                week3_max = week3_qiepian['数量_sum'].max()
                sku_qp.loc[i, 'past_3_w_sale_vol_max'] = week3_max
                week3_min = week3_qiepian['数量_sum'].min()
                sku_qp.loc[i, 'past_3_w_sale_vol_min'] = week3_min
                week3_std = np.array(week3_qiepian['数量_sum']).std()
                sku_qp.loc[i, 'past_3_w_sale_vol_std'] = week3_std
                if len(week3_qiepian) != 0:
                    week3_pay = week3_qiepian['付_款'].value_counts().index[0]
                else:
                    week3_pay = np.nan
                sku_qp.loc[i, 'past_3_w_pay_time_max'] = week3_pay

                # 对近4周数据进行切片
                week4_qiepian = sku_qp.loc[
                    (((sku_qp.loc[i, 'time'] - sku_qp['time']).apply(lambda x: x.days) <= 28) &
                     ((sku_qp.loc[i, 'time'] - sku_qp['time']).apply(lambda x: x.days) > 0))]
                week4_sum = week4_qiepian['数量_sum'].sum()
                sku_qp.loc[i, 'past_4_w_sale_vol_sum'] = week4_sum
                week4_mean = week4_qiepian['数量_sum'].mean()
                sku_qp.loc[i, 'past_4_w_sale_vol_mean'] = week4_mean
                week4_max = week4_qiepian['数量_sum'].max()
                sku_qp.loc[i, 'past_4_w_sale_vol_max'] = week4_max
                week4_min = week4_qiepian['数量_sum'].min()
                sku_qp.loc[i, 'past_4_w_sale_vol_min'] = week4_min
                week4_std = np.array(week4_qiepian['数量_sum']).std()
                sku_qp.loc[i, 'past_4_w_sale_vol_std'] = week4_std
                if len(week4_qiepian) != 0:
                    week4_pay = week4_qiepian['付_款'].value_counts().index[0]
                else:
                    week4_pay = np.nan
                sku_qp.loc[i, 'past_4_w_pay_time_max'] = week4_pay

            output = output._append(sku_qp, ignore_index=True)

        output.rename(columns={'商_品': 'item', '实际单价_min': 'item_price', '数量_sum': 'sales'}, inplace=True)
        output = output.fillna(0)

        # 因为有前30天的累计特征，所以需要把数据集最前面30天的数据都扔掉
        output.sort_values(by=['time'], ascending=True, inplace=True)
        output = output.iloc[3:] # 前三行数据日期不连续，直接扔掉
        output['diff'] = output['time'].sub(output['time'].iloc[0]).apply(lambda x: x.days)
        output['is_pre_30d'] = output['diff'] <= 30
        output = output[output['is_pre_30d'] == False]

        features = output[['sales', 'year', 'month', 'day','has_festival', 'festival_class', 'has_school_leave', 'school_leave_class',
                            'season_class', 'has_pro_exact', 'pro_exact_class', 'has_pro_range','pro_range_class','item', 'item_class', 'item_price',
                            'past_1_w_sale_vol_sum','past_2_w_sale_vol_sum', 'past_3_w_sale_vol_sum','past_4_w_sale_vol_sum',
                            'past_1_w_sale_vol_mean', 'past_2_w_sale_vol_mean','past_3_w_sale_vol_mean', 'past_4_w_sale_vol_mean',
                            'past_1_w_sale_vol_max', 'past_2_w_sale_vol_max','past_3_w_sale_vol_max', 'past_4_w_sale_vol_max',
                            'past_1_w_sale_vol_min', 'past_2_w_sale_vol_min','past_3_w_sale_vol_min', 'past_4_w_sale_vol_min',
                            'past_1_w_sale_vol_std', 'past_2_w_sale_vol_std','past_3_w_sale_vol_std', 'past_4_w_sale_vol_std',
                            'past_1_w_pay_time_max', 'past_2_w_pay_time_max','past_3_w_pay_time_max', 'past_4_w_pay_time_max']]

    elif flag == '2':
        df['商品类别'] = df['商品名称'].apply(classify)
        mappings = {'耳机':1, '屏幕':2, '平板':3, '电视机':4,'门锁':5,'电脑':6,'其他':7}
        df['商品类别'].replace(mappings, inplace=True)

        temp1 = pd.DataFrame(df[['date','商品类别','付款时段']].groupby(['date','商品类别']).agg(lambda x: x.value_counts().index[0]))
        temp = pd.DataFrame(df.groupby(['date','商品类别']).agg({'数量': 'sum', '付款时间': 'min', '实际单价': ['mean','max','min'], '商品名称':'nunique'}))
        temp = pd.concat([temp,temp1],axis=1).reset_index()

        # 重命名列名
        temp.columns = [i[0] + "_" + i[1] for i in temp.columns]
        temp.sort_values(by=['付款时间_min'], ascending=True, inplace=True)
        temp['year'] = pd.to_datetime(temp['付款时间_min']).dt.year
        temp['month'] = pd.to_datetime(temp['付款时间_min']).dt.month
        temp['day'] = pd.to_datetime(temp['付款时间_min']).dt.day

        DP = DateProcess(temp['付款时间_min'])
        df_day_label = DP.get_Feature_day()
        df_day_label['is_workday'] = df_day_label.apply(lambda x: 1 if x.day_type_chinese == '工作日' else 0, axis=1)

        df_day_label['is_festival'] = df_day_label.apply(lambda x: 0 if x.day_type_chinese == '工作日' or x.day_type_chinese == '周末或调休' else 1,axis=1)
        mappings = {'工作日': 0, '周末或调休': 0, '元旦': 1, '春节': 2, '清明节': 3, '劳动节': 4,'端午节': 5, '中秋节': 6, '国庆节': 7, '情人节': 8, '妇女节': 6, '母亲节': 7, '七夕节': 8}
        df_day_label['day_type_chinese'].replace(mappings, inplace=True)
        df_day_label.rename(columns={'day_type_chinese': 'festival_class'}, inplace=True)

        df_day_label['is_school_leave'] = df_day_label.apply(lambda x: 0 if x.school_leave == '' else 1,axis=1)
        df_day_label['school_leave'] = df_day_label.apply(lambda x: '非寒暑假' if x.school_leave == '' else x.school_leave,axis=1)
        mappings = {'非寒暑假': 0, '寒假': 1, '暑假': 2}
        df_day_label['school_leave'].replace(mappings, inplace=True)
        df_day_label.rename(columns={'school_leave': 'school_leave_class'}, inplace=True)

        mappings = {'1': 1, '2': 2, '3': 3, '4':4}
        df_day_label['quarter'].replace(mappings, inplace=True)
        df_day_label.rename(columns={'quarter': 'season_class'}, inplace=True)

        df_day_label['is_pro_exact'] = df_day_label.apply(lambda x: 0 if x.sales_type_exact == '' else 1,axis=1)
        df_day_label['sales_type_exact'] = df_day_label.apply(lambda x: '非精确促销' if x.sales_type_exact == '' else x.sales_type_exact,axis=1)
        mappings = {'非精确促销':0, '双11':1, '双12':2, '618':3}
        df_day_label['sales_type_exact'].replace(mappings, inplace=True)
        df_day_label.rename(columns={'sales_type_exact': 'pro_exact_class'}, inplace=True)

        df_day_label['is_pro_range'] = df_day_label.apply(lambda x: 0 if x.sales_type_range == '' else 1,axis=1)
        df_day_label['sales_type_range'] = df_day_label.apply(lambda x: '非范围促销' if x.sales_type_range == '' else x.sales_type_range,axis=1)
        mappings = {'非范围促销':0, '双11':1, '双12':2, '618':3}
        df_day_label['sales_type_range'].replace(mappings, inplace=True)
        df_day_label.rename(columns={'sales_type_range': 'pro_range_class'}, inplace=True)

        mappings = {'morning':1, 'afternoon':2, 'night':3, 'wee_hours':4}
        temp['付_款'].replace(mappings, inplace=True)

        temp['time'] = pd.to_datetime(temp['付款时间_min'], format='%Y-%m-%d')
        temp = pd.concat([temp,df_day_label[['is_workday','is_festival','festival_class','is_school_leave','school_leave_class','season_class','is_pro_exact','pro_exact_class','is_pro_range','pro_range_class']]],axis=1)

        # 历史累计
        output = pd.DataFrame(columns=temp.columns.tolist())

        item_cls = list(temp.loc[:,'商_品'].unique())
        for c in item_cls:
            cls_qp = temp.loc[(temp['商_品'] == c)].reset_index()
            for i in cls_qp.index:
                # 对近1天数据进行切片
                day1_qiepian = cls_qp.loc[
                    (((cls_qp.loc[i,'time']-cls_qp['time']).apply(lambda x: x.days) <= 1) &
                     ((cls_qp.loc[i,'time']-cls_qp['time']).apply(lambda x: x.days) > 0))]
                day1_sum = day1_qiepian['数量_sum'].sum()
                cls_qp.loc[i,'past_1_d_sale_vol_sum']=day1_sum

                # 对近3天数据进行切片
                day3_qiepian = cls_qp.loc[
                    (((cls_qp.loc[i,'time']-cls_qp['time']).apply(lambda x: x.days) <= 3) &
                     ((cls_qp.loc[i,'time']-cls_qp['time']).apply(lambda x: x.days) > 0))]
                day3_sum = day3_qiepian['数量_sum'].sum()
                cls_qp.loc[i,'past_3_d_sale_vol_sum']=day3_sum
                day3_mean = day3_qiepian['数量_sum'].mean()
                cls_qp.loc[i, 'past_3_d_sale_vol_mean'] = day3_mean
                day3_max = day3_qiepian['数量_sum'].max()
                cls_qp.loc[i, 'past_3_d_sale_vol_max'] = day3_max
                day3_min = day3_qiepian['数量_sum'].min()
                cls_qp.loc[i, 'past_3_d_sale_vol_min'] = day3_min
                day3_std = np.array(day3_qiepian['数量_sum']).std()
                cls_qp.loc[i, 'past_3_d_sale_vol_std'] = day3_std
                # 对7天数据进行切片
                day7_qiepian = cls_qp.loc[
                    (((cls_qp.loc[i,'time']-cls_qp['time']).apply(lambda x: x.days) <= 7) &
                     ((cls_qp.loc[i,'time']-cls_qp['time']).apply(lambda x: x.days) > 0))]
                day7_sum = day7_qiepian['数量_sum'].sum()
                cls_qp.loc[i, 'past_7_d_sale_vol_sum'] = day7_sum
                day7_mean = day7_qiepian['数量_sum'].mean()
                cls_qp.loc[i, 'past_7_d_sale_vol_mean'] = day7_mean
                day7_max = day7_qiepian['数量_sum'].max()
                cls_qp.loc[i, 'past_7_d_sale_vol_max'] = day7_max
                day7_min = day7_qiepian['数量_sum'].min()
                cls_qp.loc[i, 'past_7_d_sale_vol_min'] = day7_min
                day7_std = np.array(day7_qiepian['数量_sum']).std()
                cls_qp.loc[i, 'past_7_d_sale_vol_std'] = day7_std
                if len(day7_qiepian) != 0:
                    day7_pay = day7_qiepian['付_款'].value_counts().index[0]
                else:
                    day7_pay = np.nan
                cls_qp.loc[i, 'past_7_d_pay_time_max'] = day7_pay

                # 对30天数据进行切片
                day30_qiepian = cls_qp.loc[
                    (((cls_qp.loc[i,'time']-cls_qp['time']).apply(lambda x: x.days) <= 30) &
                     ((cls_qp.loc[i,'time']-cls_qp['time']).apply(lambda x: x.days) > 0))]
                day30_sum = day30_qiepian['数量_sum'].sum()
                cls_qp.loc[i, 'past_30_d_sale_vol_sum'] = day30_sum
                day30_mean = day30_qiepian['数量_sum'].mean()
                cls_qp.loc[i, 'past_30_d_sale_vol_mean'] = day30_mean
                day30_max = day30_qiepian['数量_sum'].max()
                cls_qp.loc[i, 'past_30_d_sale_vol_max'] = day30_max
                day30_min = day30_qiepian['数量_sum'].min()
                cls_qp.loc[i, 'past_30_d_sale_vol_min'] = day30_min
                day30_std = np.array(day30_qiepian['数量_sum']).std()
                cls_qp.loc[i, 'past_30_d_sale_vol_std'] = day30_std
                if len(day30_qiepian) != 0:
                    day30_pay = day30_qiepian['付_款'].value_counts().index[0]
                else:
                    day30_pay = np.nan
                cls_qp.loc[i, 'past_30_d_pay_time_max'] = day30_pay

            output = output._append(cls_qp, ignore_index=True)

        output.rename(columns={'商_品': 'item_class','数量_sum':'sales','实际单价_mean': 'item_price_mean','实际单价_max':'item_price_max','实际单价_min':'item_price_min','商品名称_nunique':'item_spu_num'}, inplace=True)
        output = output.fillna(0)

        # 因为有前30天的累计特征，所以需要把数据集最前面30天的数据都扔掉
        output.sort_values(by=['time'], ascending=True, inplace=True)
        output = output.iloc[3:] # 前三行数据日期不连续，直接扔掉
        output['diff'] = output['time'].sub(output['time'].iloc[0]).apply(lambda x: x.days)
        output['is_pre_30d'] = output['diff'] <= 30
        output = output[output['is_pre_30d'] == False]

        features = output[['sales','year','month','day','is_workday',
                    'is_festival','festival_class','is_school_leave','school_leave_class','season_class','is_pro_exact','pro_exact_class','is_pro_range','pro_range_class','item_class','item_price_mean','item_price_max','item_price_min','item_spu_num',
                    'past_1_d_sale_vol_sum','past_3_d_sale_vol_sum','past_7_d_sale_vol_sum','past_30_d_sale_vol_sum',
                                             'past_3_d_sale_vol_mean','past_7_d_sale_vol_mean','past_30_d_sale_vol_mean',
                                             'past_3_d_sale_vol_max','past_7_d_sale_vol_max','past_30_d_sale_vol_max',
                                             'past_3_d_sale_vol_min','past_7_d_sale_vol_min','past_30_d_sale_vol_min','past_7_d_sale_vol_std',
                                             'past_30_d_sale_vol_std','past_7_d_pay_time_max','past_30_d_pay_time_max']]

    elif flag == '3':
        df['商品类别'] = df['商品名称'].apply(classify)
        mappings = {'耳机':1, '屏幕':2, '平板':3, '电视机':4,'门锁':5,'电脑':6,'其他':7}
        df['商品类别'].replace(mappings, inplace=True)

        temp1 = pd.DataFrame(
            df[['week', '商品类别', '付款时段']].groupby(['week', '商品类别']).agg(lambda x: x.value_counts().index[0]))
        temp2 = pd.DataFrame(
            df.groupby(['week', '商品类别']).agg({'数量': 'sum', '付款时间': 'min', '实际单价': ['mean','max','min'], '商品名称':'nunique'}))
        temp3 = pd.concat([temp2, temp1], axis=1).reset_index()
        temp3.columns = [i[0] + "_" + i[1] for i in temp3.columns]
        temp3['year'] = pd.to_datetime(temp3['付款时间_min']).dt.year
        temp3['month'] = pd.to_datetime(temp3['付款时间_min']).dt.month
        temp3['day'] = pd.to_datetime(temp3['付款时间_min']).dt.day

        mappings = {'morning': 1, 'afternoon': 2, 'night': 3, 'wee_hours': 4}
        temp3['付_款'].replace(mappings, inplace=True)

        temp = pd.DataFrame(
            df.groupby(['date', '商品类别']).agg({'付款时间': {'min'},'week': {'first'}}))
        temp.columns = [i[0] + "_" + i[1] for i in temp.columns]
        DP = DateProcess(temp['付款时间_min'])
        df_day_label = DP.get_Feature_day()
        df_day_label['is_workday'] = df_day_label.apply(lambda x: 1 if x.day_type_chinese == '工作日' else 0, axis=1)
        df_day_label['is_festival'] = df_day_label.apply(
            lambda x: 0 if x.day_type_chinese == '工作日' or x.day_type_chinese == '周末或调休' else 1, axis=1)
        mappings = {'工作日': 0, '周末或调休': 0, '元旦': 1, '春节': 2, '清明节': 3, '劳动节': 4, '端午节': 5,
                    '中秋节': 6, '国庆节': 7, '情人节': 8, '妇女节': 6, '母亲节': 7, '七夕节': 8}
        df_day_label['day_type_chinese'].replace(mappings, inplace=True)
        df_day_label.rename(columns={'day_type_chinese': 'festival_class'}, inplace=True)

        df_day_label['is_school_leave'] = df_day_label.apply(lambda x: 0 if x.school_leave == '' else 1, axis=1)
        df_day_label['school_leave'] = df_day_label.apply(
            lambda x: '非寒暑假' if x.school_leave == '' else x.school_leave, axis=1)
        mappings = {'非寒暑假': 0, '寒假': 1, '暑假': 2}
        df_day_label['school_leave'].replace(mappings, inplace=True)
        df_day_label.rename(columns={'school_leave': 'school_leave_class'}, inplace=True)

        mappings = {'1': 1, '2': 2, '3': 3, '4': 4}
        df_day_label['quarter'].replace(mappings, inplace=True)
        df_day_label.rename(columns={'quarter': 'season_class'}, inplace=True)

        df_day_label['is_pro_exact'] = df_day_label.apply(lambda x: 0 if x.sales_type_exact == '' else 1, axis=1)
        df_day_label['sales_type_exact'] = df_day_label.apply(
            lambda x: '非精确促销' if x.sales_type_exact == '' else x.sales_type_exact, axis=1)
        mappings = {'非精确促销': 0, '双11': 1, '双12': 2, '618': 3}
        df_day_label['sales_type_exact'].replace(mappings, inplace=True)
        df_day_label.rename(columns={'sales_type_exact': 'pro_exact_class'}, inplace=True)

        df_day_label['is_pro_range'] = df_day_label.apply(lambda x: 0 if x.sales_type_range == '' else 1, axis=1)
        df_day_label['sales_type_range'] = df_day_label.apply(
            lambda x: '非范围促销' if x.sales_type_range == '' else x.sales_type_range, axis=1)
        mappings = {'非范围促销': 0, '双11': 1, '双12': 2, '618': 3}
        df_day_label['sales_type_range'].replace(mappings, inplace=True)
        df_day_label.rename(columns={'sales_type_range': 'pro_range_class'}, inplace=True)

        temp = pd.concat([temp, df_day_label], axis=1)

        temp4 = temp.groupby(['week_first', '商品类别']).apply(date_2_week).reset_index()

        temp3['time'] = pd.to_datetime(temp3['付款时间_min'], format='%Y-%m-%d')
        temp3 = pd.concat([temp3, temp4], axis=1)

        # 历史累计
        output = pd.DataFrame(columns=temp3.columns.tolist())

        item_cls = list(temp3.loc[:, '商_品'].unique())
        for c in item_cls:
            cls_qp = temp3.loc[(temp3['商_品'] == c)].reset_index()
            for i in cls_qp.index:
                # 对近1周数据进行切片
                week1_qiepian = cls_qp.loc[
                    (((cls_qp.loc[i, 'time'] - cls_qp['time']).apply(lambda x: x.days) <= 7) &
                     ((cls_qp.loc[i, 'time'] - cls_qp['time']).apply(lambda x: x.days) > 0))]
                week1_sum = week1_qiepian['数量_sum'].sum()
                cls_qp.loc[i, 'past_1_w_sale_vol_sum'] = week1_sum
                week1_mean = week1_qiepian['数量_sum'].mean()
                cls_qp.loc[i, 'past_1_w_sale_vol_mean'] = week1_mean
                week1_max = week1_qiepian['数量_sum'].max()
                cls_qp.loc[i, 'past_1_w_sale_vol_max'] = week1_max
                week1_min = week1_qiepian['数量_sum'].min()
                cls_qp.loc[i, 'past_1_w_sale_vol_min'] = week1_min
                week1_std = np.array(week1_qiepian['数量_sum']).std()
                cls_qp.loc[i, 'past_1_w_sale_vol_std'] = week1_std
                if len(week1_qiepian) != 0:
                    week1_pay = week1_qiepian['付_款'].value_counts().index[0]
                else:
                    week1_pay = np.nan
                cls_qp.loc[i, 'past_1_w_pay_time_max'] = week1_pay

                # 对近2周数据进行切片
                week2_qiepian = cls_qp.loc[
                    (((cls_qp.loc[i, 'time'] - cls_qp['time']).apply(lambda x: x.days) <= 14) &
                     ((cls_qp.loc[i, 'time'] - cls_qp['time']).apply(lambda x: x.days) > 0))]
                week2_sum = week2_qiepian['数量_sum'].sum()
                cls_qp.loc[i, 'past_2_w_sale_vol_sum'] = week2_sum
                week2_mean = week2_qiepian['数量_sum'].mean()
                cls_qp.loc[i, 'past_2_w_sale_vol_mean'] = week2_mean
                week2_max = week2_qiepian['数量_sum'].max()
                cls_qp.loc[i, 'past_2_w_sale_vol_max'] = week2_max
                week2_min = week2_qiepian['数量_sum'].min()
                cls_qp.loc[i, 'past_2_w_sale_vol_min'] = week2_min
                week2_std = np.array(week2_qiepian['数量_sum']).std()
                cls_qp.loc[i, 'past_2_w_sale_vol_std'] = week2_std
                if len(week2_qiepian) != 0:
                    week2_pay = week2_qiepian['付_款'].value_counts().index[0]
                else:
                    week2_pay = np.nan
                cls_qp.loc[i, 'past_2_w_pay_time_max'] = week2_pay

                # 对近3周数据进行切片
                week3_qiepian = cls_qp.loc[
                    (((cls_qp.loc[i, 'time'] - cls_qp['time']).apply(lambda x: x.days) <= 21) &
                     ((cls_qp.loc[i, 'time'] - cls_qp['time']).apply(lambda x: x.days) > 0))]
                week3_sum = week3_qiepian['数量_sum'].sum()
                cls_qp.loc[i, 'past_3_w_sale_vol_sum'] = week3_sum
                week3_mean = week3_qiepian['数量_sum'].mean()
                cls_qp.loc[i, 'past_3_w_sale_vol_mean'] = week3_mean
                week3_max = week3_qiepian['数量_sum'].max()
                cls_qp.loc[i, 'past_3_w_sale_vol_max'] = week3_max
                week3_min = week3_qiepian['数量_sum'].min()
                cls_qp.loc[i, 'past_3_w_sale_vol_min'] = week3_min
                week3_std = np.array(week3_qiepian['数量_sum']).std()
                cls_qp.loc[i, 'past_3_w_sale_vol_std'] = week3_std
                if len(week3_qiepian) != 0:
                    week3_pay = week3_qiepian['付_款'].value_counts().index[0]
                else:
                    week3_pay = np.nan
                cls_qp.loc[i, 'past_3_w_pay_time_max'] = week3_pay

                # 对近4周数据进行切片
                week4_qiepian = cls_qp.loc[
                    (((cls_qp.loc[i, 'time'] - cls_qp['time']).apply(lambda x: x.days) <= 28) &
                     ((cls_qp.loc[i, 'time'] - cls_qp['time']).apply(lambda x: x.days) > 0))]
                week4_sum = week4_qiepian['数量_sum'].sum()
                cls_qp.loc[i, 'past_4_w_sale_vol_sum'] = week4_sum
                week4_mean = week4_qiepian['数量_sum'].mean()
                cls_qp.loc[i, 'past_4_w_sale_vol_mean'] = week4_mean
                week4_max = week4_qiepian['数量_sum'].max()
                cls_qp.loc[i, 'past_4_w_sale_vol_max'] = week4_max
                week4_min = week4_qiepian['数量_sum'].min()
                cls_qp.loc[i, 'past_4_w_sale_vol_min'] = week4_min
                week4_std = np.array(week4_qiepian['数量_sum']).std()
                cls_qp.loc[i, 'past_4_w_sale_vol_std'] = week4_std
                if len(week4_qiepian) != 0:
                    week4_pay = week4_qiepian['付_款'].value_counts().index[0]
                else:
                    week4_pay = np.nan
                cls_qp.loc[i, 'past_4_w_pay_time_max'] = week4_pay

            output = output._append(cls_qp, ignore_index=True)

        output.rename(columns={'商_品': 'item_class','实际单价_mean': 'item_price_mean','实际单价_max': 'item_price_max','实际单价_min': 'item_price_min','商品名称_nunique':'item_spu_num', '数量_sum': 'sales'}, inplace=True)
        output = output.fillna(0)

        # 因为有前30天的累计特征，所以需要把数据集最前面30天的数据都扔掉
        output.sort_values(by=['time'], ascending=True, inplace=True)
        output = output.iloc[3:] # 前三行数据日期不连续，直接扔掉
        output['diff'] = output['time'].sub(output['time'].iloc[0]).apply(lambda x: x.days)
        output['is_pre_30d'] = output['diff'] <= 30
        output = output[output['is_pre_30d'] == False]

        features = output[['sales', 'year', 'month', 'day','has_festival', 'festival_class', 'has_school_leave', 'school_leave_class',
                            'season_class', 'has_pro_exact', 'pro_exact_class', 'has_pro_range','pro_range_class','item_class', 'item_price_mean','item_price_max','item_price_min','item_spu_num',
                            'past_1_w_sale_vol_sum','past_2_w_sale_vol_sum', 'past_3_w_sale_vol_sum','past_4_w_sale_vol_sum',
                            'past_1_w_sale_vol_mean', 'past_2_w_sale_vol_mean','past_3_w_sale_vol_mean', 'past_4_w_sale_vol_mean',
                            'past_1_w_sale_vol_max', 'past_2_w_sale_vol_max','past_3_w_sale_vol_max', 'past_4_w_sale_vol_max',
                            'past_1_w_sale_vol_min', 'past_2_w_sale_vol_min','past_3_w_sale_vol_min', 'past_4_w_sale_vol_min',
                            'past_1_w_sale_vol_std', 'past_2_w_sale_vol_std','past_3_w_sale_vol_std', 'past_4_w_sale_vol_std',
                            'past_1_w_pay_time_max', 'past_2_w_pay_time_max','past_3_w_pay_time_max', 'past_4_w_pay_time_max']]

    return features



if __name__ == '__main__':
    # load data

    ori_df = pd.read_excel('************\\data\\华为发货明细23.6.1-24.6.2.xlsx')
    output_dir = '***********\\result\\jiadian'

    print(len(ori_df.index.values)) # records num
    ori_df.head() # show head data

    ori_df = ori_df.replace(0, np.nan)  # 把0替换成nan
    ori_df = ori_df.replace('', np.nan)  # 把''替换成nan
    df = ori_df[['商品名称', '数量', '实际单价', '付款时间']].dropna(axis=0, how='any')  # choose fields and drop null
    df.drop_duplicates(inplace=True)
    print(len(df.index.values))  # records num after filtering

    # feature processing
    name = {'0': 'spu粒度（天）', '1': 'spu粒度（周）', '2': '商品大类粒度（天）', '3': '商品大类粒度（周）'}
    flag = '1'

    features = feature_process(df, flag)

    # feature save as file
    writer = pd.ExcelWriter(output_dir + '\\' + name[flag] + "_features.xlsx")
    features.to_excel(writer)
    writer._save()
    writer.close()

    # feature relevance
    if flag in ['0','1']:
        features = features.drop(['item'], axis=1)
    plt.subplots(figsize=(24, 20))
    sns.heatmap(features.corr(numeric_only=True), cmap='RdYlGn', annot=True, vmin=-0.1, vmax=0.1, center=0)
    plt.savefig(output_dir + '\\' + name[flag] +'特征相关性.png', bbox_inches='tight')
    plt.show()

