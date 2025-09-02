import re

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

input_dir = '***********\\SalesForecast\\data'
output_dir = '****************\\SalesForecast\\result\\meizhuang'


def classify(x):

    cls1 = ['面膜','眼膜','唇膜']
    cls2 = ['蜜粉','散粉']
    cls3 = ['气垫BB','BB霜']
    cls4 = ['安瓶','精华','乳液','面霜','晚霜','保湿乳','保湿液','黑霜']
    cls5 = ['粉扑','美妆蛋','洗脸扑','海绵']
    cls6 = ['身体乳']
    cls7 = ['高光','阴影','修容膏','修容笔']
    cls8 = ['眉笔','眉粉','眉膏']
    cls9 = ['隔离霜', '妆前乳','妆前霜','隔离乳']
    cls10 = ['粉底液', '粉底膏', '粉底','粉膏']
    cls11 = ['卸妆油','卸妆水','卸妆液']
    cls12 = ['眼影']
    cls13 = ['护手霜']
    cls14 = ['粉饼']
    cls15 = ['腮红']
    cls16 = ['遮瑕膏','遮瑕液','遮瑕笔']
    cls17 = ['眼线笔','眼线液','眼线膏']
    cls18 = ['化妆刷','工具刷','鼻影刷','遮瑕刷']

    if any([keyword in x for keyword in cls1]):
        x = '面膜/眼膜/唇膜'
    elif any([keyword in x for keyword in cls2]):
        x = '蜜粉/散粉'
    elif any([keyword in x for keyword in cls3]):
        x = '气垫BB/BB霜'
    elif any([keyword in x for keyword in cls4]):
        x = '面部护肤'
    elif any([keyword in x for keyword in cls5]):
        x = '粉扑'
    elif any([keyword in x for keyword in cls6]):
        x = '身体护肤'
    elif any([keyword in x for keyword in cls8]):
        x = '眉笔/眉粉/眉膏'
    elif any([keyword in x for keyword in cls9]):
        x = '隔离霜/妆前乳'
    elif any([keyword in x for keyword in cls10]):
        x = '粉底液/膏'
    elif any([keyword in x for keyword in cls17]):
        x = '眼线笔'
    elif any([keyword in x for keyword in cls15]):
        x = '腮红'
    elif any([keyword in x for keyword in cls12]):
        x = '眼影'
    elif any([keyword in x for keyword in cls13]):
        x = '手部护肤'
    elif any([keyword in x for keyword in cls18]):
        x = '化妆刷'
    elif any([keyword in x for keyword in cls14]):
        x = '粉饼'
    elif any([keyword in x for keyword in cls11]):
        x = '卸妆液/油/水'
    elif any([keyword in x for keyword in cls7]):
        x = '修容膏/笔'
    elif any([keyword in x for keyword in cls16]):
        x = '遮瑕膏/笔'
    else:
        x = '其他'
    return x

def item_info(x):
    weight_or_volume = 999.0 # default
    extraction = re.findall(r'([0-9.]{,4})g',x) + re.findall(r'([0-9.]{,4})ml',x)
    if extraction:
        weight_or_volume = float(extraction[0])

    return weight_or_volume

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

    df['date'] = pd.to_datetime(df['付款确认时间']).dt.day.apply(str).str.cat(
        pd.to_datetime(df['付款确认时间']).dt.month.apply(str), sep=',').str.cat(
        pd.to_datetime(df['付款确认时间']).dt.year.apply(str), sep=',')

    # 付款时段
    df['付款时段'] = df['付款确认时间'].apply(pay_range)
    features = None

    temp1 = pd.DataFrame(df[['date','商品名称','付款时段']].groupby(['date','商品名称']).agg(lambda x: x.value_counts().index[0]))
    temp = pd.DataFrame(df.groupby(['date','商品名称']).agg({'商品ID': {'max'},'订购数量': {'sum'}, '付款确认时间': {'min'}, '实际单价': {'min'}}))
    temp = pd.concat([temp,temp1],axis=1).reset_index()

    # 重命名列名
    temp.columns = [i[0] + "_" + i[1] for i in temp.columns]
    temp.sort_values(by=['付款确认时间_min'], ascending=True, inplace=True)
    temp['year'] = pd.to_datetime(temp['付款确认时间_min']).dt.year
    temp['month'] = pd.to_datetime(temp['付款确认时间_min']).dt.month
    temp['day'] = pd.to_datetime(temp['付款确认时间_min']).dt.day

    DP = DateProcess(temp['付款确认时间_min'])
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

    temp['商品类别'] = temp['商_品'].apply(classify)
    mappings = {'面膜/眼膜/唇膜':1, '蜜粉/散粉':2, '气垫BB/BB霜':3, '面部护肤':4,'粉扑':5,'身体护肤':6,'眉笔/眉粉/眉膏':7,'隔离霜/妆前乳':8,'粉底液/膏':9,'眼线笔':10,'腮红':11,'眼影':12,'手部护肤':13,'化妆刷':14,'粉饼':15,'卸妆液/油/水':16,'修容膏/笔':17,'遮瑕膏/笔':18,'其他':19}
    temp['商品类别'].replace(mappings, inplace=True)
    temp.rename(columns={'商品类别': 'item_class'}, inplace=True)

    mappings = {'morning':1, 'afternoon':2, 'night':3, 'wee_hours':4}
    temp['付_款'].replace(mappings, inplace=True)

    temp['time'] = pd.to_datetime(temp['付款确认时间_min'])
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
            day1_sum = day1_qiepian['订购数量_sum'].sum()
            sku_qp.loc[i,'past_1_d_sale_vol_sum']=day1_sum

            # 对近3天数据进行切片
            day3_qiepian = sku_qp.loc[
                (((sku_qp.loc[i,'time']-sku_qp['time']).apply(lambda x: x.days) <= 3) &
                 ((sku_qp.loc[i,'time']-sku_qp['time']).apply(lambda x: x.days) > 0))]
            day3_sum = day3_qiepian['订购数量_sum'].sum()
            sku_qp.loc[i,'past_3_d_sale_vol_sum']=day3_sum
            day3_mean = day3_qiepian['订购数量_sum'].mean()
            sku_qp.loc[i, 'past_3_d_sale_vol_mean'] = day3_mean
            day3_max = day3_qiepian['订购数量_sum'].max()
            sku_qp.loc[i, 'past_3_d_sale_vol_max'] = day3_max
            day3_min = day3_qiepian['订购数量_sum'].min()
            sku_qp.loc[i, 'past_3_d_sale_vol_min'] = day3_min
            day3_std = np.array(day3_qiepian['订购数量_sum']).std()
            sku_qp.loc[i, 'past_3_d_sale_vol_std'] = day3_std
            # 对7天数据进行切片
            day7_qiepian = sku_qp.loc[
                (((sku_qp.loc[i,'time']-sku_qp['time']).apply(lambda x: x.days) <= 7) &
                 ((sku_qp.loc[i,'time']-sku_qp['time']).apply(lambda x: x.days) > 0))]
            day7_sum = day7_qiepian['订购数量_sum'].sum()
            sku_qp.loc[i, 'past_7_d_sale_vol_sum'] = day7_sum
            day7_mean = day7_qiepian['订购数量_sum'].mean()
            sku_qp.loc[i, 'past_7_d_sale_vol_mean'] = day7_mean
            day7_max = day7_qiepian['订购数量_sum'].max()
            sku_qp.loc[i, 'past_7_d_sale_vol_max'] = day7_max
            day7_min = day7_qiepian['订购数量_sum'].min()
            sku_qp.loc[i, 'past_7_d_sale_vol_min'] = day7_min
            day7_std = np.array(day7_qiepian['订购数量_sum']).std()
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
            day30_sum = day30_qiepian['订购数量_sum'].sum()
            sku_qp.loc[i, 'past_30_d_sale_vol_sum'] = day30_sum
            day30_mean = day30_qiepian['订购数量_sum'].mean()
            sku_qp.loc[i, 'past_30_d_sale_vol_mean'] = day30_mean
            day30_max = day30_qiepian['订购数量_sum'].max()
            sku_qp.loc[i, 'past_30_d_sale_vol_max'] = day30_max
            day30_min = day30_qiepian['订购数量_sum'].min()
            sku_qp.loc[i, 'past_30_d_sale_vol_min'] = day30_min
            day30_std = np.array(day30_qiepian['订购数量_sum']).std()
            sku_qp.loc[i, 'past_30_d_sale_vol_std'] = day30_std
            if len(day30_qiepian) != 0:
                day30_pay = day30_qiepian['付_款'].value_counts().index[0]
            else:
                day30_pay = np.nan
            sku_qp.loc[i, 'past_30_d_pay_time_max'] = day30_pay

        output = output._append(sku_qp, ignore_index=True)

    output.rename(columns={'商品ID_max':'商品ID', '商_品': 'item','实际单价_min':'item_price','订购数量_sum':'sales'}, inplace=True)
    output = output.fillna(0)

    # 因为有前30天的累计特征，所以需要把数据集最前面30天的数据都扔掉
    output.sort_values(by=['time'], ascending=True, inplace=True)
    #output = output.iloc[3:] # 日期不连续，直接扔掉
    output['diff'] = output['time'].sub(output['time'].iloc[0]).apply(lambda x: x.days)
    output['is_pre_30d'] = output['diff'] <= 30
    output = output[output['is_pre_30d'] == False]

    # 商品重量体积
    output['item_weight_volume'] = output['item'].apply(item_info)
    # 营销
    marketing_df = pd.read_excel(input_dir + '\\' + '营销SKU粒度_计划安排.xlsx')
    marketing_df = marketing_df.replace('', np.nan)  # 把''替换成nan
    marketing_df = marketing_df[['SKU ID', '点击时间', '花费', '展现数']].dropna(axis=0,how='any')  # choose fields and drop null
    marketing_df.drop_duplicates(inplace=True)
    marketing_df.rename(columns={"SKU ID": "商品ID"},inplace=True)
    marketing_df['time_str'] = marketing_df['点击时间'].str[:8]
    bins = np.linspace(marketing_df["花费"].min(), marketing_df["花费"].max(), 6)
    marketing_df['cost_level'] = pd.cut(x=marketing_df["花费"],bins=bins,labels=[1,2,3,4,5])
    marketing_df['cost_level'] = marketing_df['cost_level'].cat.add_categories([0])
    marketing_df['cost_level'] = marketing_df['cost_level'].fillna(0).astype('int')
    bins = np.linspace(marketing_df["展现数"].min(), marketing_df["展现数"].max(), 6)
    marketing_df['show_level'] = pd.cut(x=marketing_df["展现数"],bins=bins,labels=[1,2,3,4,5])
    marketing_df['show_level'] = marketing_df['show_level'].cat.add_categories([0])
    marketing_df['show_level'] = marketing_df['show_level'].fillna(0).astype('int')
    marketing_df['marketing_level'] = marketing_df[['cost_level', 'show_level']].max(axis=1)
    output['time_str'] = output['time'].dt.strftime('%Y%m%d')
    output = pd.merge(output, marketing_df, on=['商品ID','time_str'], how='left')
    output['is_marketing'] = output['cost_level'].map(lambda x: 0 if np.isnan(x) else 1)

    # 单品促销
    pro_df1 = pd.read_excel(input_dir + '\\' + '营销工具活动效果_单品促销SKU粒度_实际效果.xls',sheet_name='活动效果榜单')
    pro_df2 = pd.read_excel(input_dir + '\\' + '营销工具活动效果_单品促销SKU粒度_实际效果.xls',sheet_name='促销明细')

    pro_df1 = pro_df1.replace('', np.nan)
    pro_df1 = pro_df1[['促销id', '开始时间', '结束时间']].dropna(axis=0,how='any')  # choose fields and drop null
    pro_df1.drop_duplicates(inplace=True)

    pro_df2 = pro_df2.replace('', np.nan)  # 把''替换成nan
    pro_df2 = pro_df2[['促销id', 'SKUID']].dropna(axis=0,how='any')  # choose fields and drop null
    pro_df2.rename(columns={"SKUID": "商品ID"}, inplace=True)
    pro_df2.drop_duplicates(inplace=True)

    pro_df1 = pd.merge(pro_df1, pro_df2, how='inner', on='促销id')
    pro_df1['time_str'] = pro_df1.apply(lambda x: pd.date_range(x['开始时间'],x['结束时间']).tolist(),axis=1)
    pro_df1 = pro_df1.explode('time_str')
    pro_df1['time_str'] = pro_df1['time_str'].dt.strftime('%Y%m%d')

    output = pd.merge(output, pro_df1, on=['商品ID','time_str'], how='left')
    output['is_pro_single_item'] = output['促销id'].map(lambda x: 0 if np.isnan(x) else 1)

    # 优惠券促销
    # pro_df1 = pd.read_excel(input_dir + '\\' + '营销工具活动效果_优惠券促销SKU粒度_计划安排.xlsx',sheet_name='商品券')
    # pro_df2 = pd.read_excel(input_dir + '\\' + '营销工具活动效果_优惠券促销SKU粒度_计划安排.xlsx',sheet_name='店铺券')
    #
    # pro_df1 = pro_df1.replace('', np.nan)  # 把''替换成nan
    # pro_df1 = pro_df1[['领券开始时间', '领券结束时间', '发放数量(张)','适用SKU']].dropna(axis=0,how='any')  # choose fields and drop null
    # pro_df1.rename(columns={"适用SKU": "商品ID"}, inplace=True)
    # pro_df1.drop_duplicates(inplace=True)
    # pro_df1['time_str'] = pro_df1.apply(lambda x: pd.date_range(x['领券开始时间'],x['领券结束时间']).tolist(),axis=1)
    # pro_df1 = pro_df1.explode('time_str')
    # pro_df1['time_str'] = pro_df1['time_str'].dt.strftime('%Y%m%d')
    # output = pd.merge(output, pro_df1, on=['商品ID','time_str'], how='left')
    # output['is_pro_item_voucher'] = output['发放数量(张)'].map(lambda x: 0 if np.isnan(x) else 1)
    # output.rename(columns={"发放数量(张)": "item_voucher_num"}, inplace=True)
    # output['item_voucher_num'] = output['item_voucher_num'].replace(np.nan, 0)
    #
    # pro_df2 = pro_df2.replace('', np.nan)  # 把''替换成nan
    # pro_df2 = pro_df2[['领券开始时间', '领券结束时间', '发放数量(张)']].dropna(axis=0,how='any')  # choose fields and drop null
    # pro_df2.drop_duplicates(inplace=True)
    # pro_df2['time_str'] = pro_df2.apply(lambda x: pd.date_range(x['领券开始时间'],x['领券结束时间']).tolist(),axis=1)
    # pro_df2 = pro_df2.explode('time_str')
    # pro_df2['time_str'] = pro_df2['time_str'].dt.strftime('%Y%m%d')
    # output = pd.merge(output, pro_df2, on=['time_str'], how='left')
    # output['is_pro_store_voucher'] = output['发放数量(张)'].map(lambda x: 0 if np.isnan(x) else 1)
    # output.rename(columns={"发放数量(张)": "store_voucher_num"}, inplace=True)
    # output['store_voucher_num'] = output['store_voucher_num'].replace(np.nan, 0)

    if flag == 'basic_feats':
        features = output[['sales','year','month','day','is_workday',
                'is_festival','festival_class','is_school_leave','school_leave_class','season_class','is_pro_exact','pro_exact_class','is_pro_range','pro_range_class','item','item_class','item_price',
                'past_1_d_sale_vol_sum','past_3_d_sale_vol_sum','past_7_d_sale_vol_sum','past_30_d_sale_vol_sum',
                                         'past_3_d_sale_vol_mean','past_7_d_sale_vol_mean','past_30_d_sale_vol_mean',
                                         'past_3_d_sale_vol_max','past_7_d_sale_vol_max','past_30_d_sale_vol_max',
                                         'past_3_d_sale_vol_min','past_7_d_sale_vol_min','past_30_d_sale_vol_min','past_7_d_sale_vol_std',
                                         'past_30_d_sale_vol_std','past_7_d_pay_time_max','past_30_d_pay_time_max']]
        features.drop_duplicates(inplace=True)

    elif flag == 'additional_feats':
        features = output[['sales','year','month','day','is_workday',
                'is_festival','festival_class','is_school_leave','school_leave_class','season_class','is_pro_exact','pro_exact_class','is_pro_range','pro_range_class','item','item_class','item_price',
                'past_1_d_sale_vol_sum','past_3_d_sale_vol_sum','past_7_d_sale_vol_sum','past_30_d_sale_vol_sum',
                                         'past_3_d_sale_vol_mean','past_7_d_sale_vol_mean','past_30_d_sale_vol_mean',
                                         'past_3_d_sale_vol_max','past_7_d_sale_vol_max','past_30_d_sale_vol_max',
                                         'past_3_d_sale_vol_min','past_7_d_sale_vol_min','past_30_d_sale_vol_min','past_7_d_sale_vol_std',
                                         'past_30_d_sale_vol_std','past_7_d_pay_time_max','past_30_d_pay_time_max',
                'item_weight_volume','is_marketing','marketing_level','is_pro_single_item']]
        features.drop_duplicates(inplace=True)

    elif flag == 'optimized_feats':
        features = output[['sales','month','day','is_workday',
                'is_pro_exact','item','item_class','item_price',
                'past_1_d_sale_vol_sum','past_3_d_sale_vol_sum','past_7_d_sale_vol_sum','past_30_d_sale_vol_sum',
                                         'past_3_d_sale_vol_mean','past_7_d_sale_vol_mean','past_30_d_sale_vol_mean',
                                         'past_3_d_sale_vol_max','past_7_d_sale_vol_max','past_30_d_sale_vol_max',
                                         'past_3_d_sale_vol_min','past_7_d_sale_vol_min','past_30_d_sale_vol_min','past_7_d_sale_vol_std',
                                         'past_30_d_sale_vol_std',
                'item_weight_volume','marketing_level','is_pro_single_item']]
        features.drop_duplicates(inplace=True)

    return features



if __name__ == '__main__':
    # load data

    ori_df = pd.read_excel(input_dir + '\\' + '合并_毛戈平京东pop订单_to0628.xlsx')

    print(len(ori_df.index.values)) # records num
    ori_df.head() # show head data

    ori_df = ori_df.replace(0, np.nan)  # 把0替换成nan
    ori_df = ori_df.replace('', np.nan)  # 把''替换成nan
    df = ori_df[['商品ID', '商品名称', '订购数量', '订单金额', '付款确认时间']].dropna(axis=0, how='any')  # choose fields and drop null
    df.drop_duplicates(inplace=True)
    df['实际单价'] = df['订单金额'] / df['订购数量']
    # 根据商品ID，将商品SKU的类别联系起来
    # cls_df = pd.read_excel('C:\\Users\\Vivia\\工作\\code\\SalesForecast\\data\\sku_info_all.xlsx')
    # cls_df.rename(columns={'SKUID': '商品ID'}, inplace=True)
    # cls_df = cls_df[['商品ID','一级类目','二级类目','三级类目']]
    # cls_df.drop_duplicates(inplace=True)
    # df = pd.merge(df, cls_df, how='inner', on='商品ID')
    # df.drop_duplicates(inplace=True)
    print(len(df.index.values))  # records num after filtering
    print(df['订购数量'].sum())
    print(df['订单金额'].sum())

    # feature processing
    name = 'spu粒度（天）'
    flag = 'optimized_feats' # ['basic_feats','additional_feats','optimized_feats']

    features = feature_process(df, flag)

    # feature save as file
    writer = pd.ExcelWriter(output_dir + '\\' + name + "_features.xlsx")
    features.to_excel(writer)
    writer._save()
    writer.close()

    # feature relevance
    features = features.drop(['item'], axis=1)
    plt.subplots(figsize=(24, 20))
    sns.heatmap(features.corr(numeric_only=True), cmap='RdYlGn', annot=True, vmin=-0.1, vmax=0.1, center=0)
    plt.savefig(output_dir + '\\' + name +'特征相关性.png', bbox_inches='tight')
    plt.show()

