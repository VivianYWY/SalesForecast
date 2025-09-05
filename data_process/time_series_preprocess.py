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

def classify_meizhuang(x):
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


def classify_jiadian(x):
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

# 缺省值处理，时间排序
# load data
ori_df = pd.read_excel('********\\data\\合并_毛戈平京东pop订单.xlsx')
output_dir = '**********\\data'
flag = 'meizhuang'

if flag == 'meizhuang':
    print(len(ori_df.index.values)) # records num
    ori_df.head() # show head data
    ori_df = ori_df.replace(0, np.nan)  # 把0替换成nan
    ori_df = ori_df.replace('', np.nan)  # 把''替换成nan
    df = ori_df[['商品名称','订购数量','订单金额','付款确认时间']].dropna(axis=0,how='any') # choose fields and drop null
    df.drop_duplicates()
    print(len(df.index.values)) # records num after filtering

    df['date'] = pd.to_datetime(df['付款确认时间']).dt.day.apply(str).str.cat(pd.to_datetime(df['付款确认时间']).dt.month.apply(str),sep=',').str.cat(pd.to_datetime(df['付款确认时间']).dt.year.apply(str), sep=',')
    df['week'] = pd.to_datetime(df['付款确认时间']).dt.isocalendar().week.apply(str).str.cat(pd.to_datetime(df['付款确认时间']).dt.year.apply(str),sep=',')

    # day
    temp = pd.DataFrame(df.groupby('date').agg({'订购数量':{'sum'},'付款确认时间':{'min'}}))
    # 重命名列名
    temp.columns = [i[0] + "_" + i[1] for i in temp.columns]
    temp.sort_values(by=['付款确认时间_min'],ascending=True,inplace=True)
    temp['time'] = temp['付款确认时间_min'].apply(lambda x: pd.to_datetime(x).strftime('%Y-%m-%d'))
    temp.rename(columns={'订购数量_sum': 'data'}, inplace=True)

    # save as excel
    writer = pd.ExcelWriter(output_dir + '\\' + "meizhuang_time_series_data.xlsx")
    temp[['data','time']].to_excel(writer,sheet_name='day',index=False)

    # week
    temp1 = pd.DataFrame(df.groupby('week').agg({'订购数量':{'sum'},'付款确认时间':{'min'}}))
    # 重命名列名
    temp1.columns = [i[0] + "_" + i[1] for i in temp1.columns]
    temp1.sort_values(by=['付款确认时间_min'],ascending=True,inplace=True)
    temp1['time'] = temp1['付款确认时间_min'].apply(lambda x: pd.to_datetime(x).strftime('%Y-%m-%d'))
    temp1.rename(columns={'订购数量_sum': 'data'}, inplace=True)

    # save as excel
    temp1[['data','time']].to_excel(writer,sheet_name='week',index=False)

    # 销量top2的商品大类的day日销量序列（没有销量的日期，销量补充为0）
    df['商品类别'] = df['商品名称'].apply(classify_meizhuang)
    temp2 = pd.DataFrame(df[df['商品类别'] == '粉底液/膏'].groupby('date').agg({'订购数量': {'sum'}, '付款确认时间': {'min'}}))
    temp2.columns = [i[0] + "_" + i[1] for i in temp2.columns]
    temp2.sort_values(by=['付款确认时间_min'],ascending=True,inplace=True)

    temp3 = pd.DataFrame(df[df['商品类别'] == '卸妆液/油/水'].groupby('date').agg({'订购数量': {'sum'}, '付款确认时间': {'min'}}))
    temp3.columns = [i[0] + "_" + i[1] for i in temp3.columns]
    temp3.sort_values(by=['付款确认时间_min'],ascending=True,inplace=True)

    temp11 = pd.merge(temp, temp2, how='left', left_index=True, right_index=True).fillna(0)
    temp22 = pd.merge(temp, temp3, how='left', left_index=True, right_index=True).fillna(0)

    # save as excel
    output = temp11[['订购数量_sum','time']]
    output.rename(columns={'订购数量_sum': 'data'}, inplace=True)
    output.to_excel(writer,sheet_name='fendi',index=False)

    output = temp22[['订购数量_sum','time']]
    output.rename(columns={'订购数量_sum': 'data'}, inplace=True)
    output.to_excel(writer,sheet_name='xiezhuang',index=False)

    writer._save()
    writer.close()

elif flag == 'jiadian':
    print(len(ori_df.index.values))  # records num
    ori_df.head()  # show head data
    ori_df = ori_df.replace(0, np.nan)  # 把0替换成nan
    ori_df = ori_df.replace('', np.nan)  # 把''替换成nan
    df = ori_df[['商品名称', '数量', '实际单价', '付款时间']].dropna(axis=0,how='any')  # choose fields and drop null
    df.drop_duplicates()
    print(len(df.index.values))  # records num after filtering

    df['date'] = pd.to_datetime(df['付款时间']).dt.day.apply(str).str.cat(
        pd.to_datetime(df['付款时间']).dt.month.apply(str), sep=',').str.cat(
        pd.to_datetime(df['付款时间']).dt.year.apply(str), sep=',')
    df['week'] = pd.to_datetime(df['付款时间']).dt.isocalendar().week.apply(str).str.cat(
        pd.to_datetime(df['付款时间']).dt.year.apply(str), sep=',')

    # day
    temp = pd.DataFrame(df.groupby('date').agg({'数量': {'sum'}, '付款时间': {'min'}}))
    # 重命名列名
    temp.columns = [i[0] + "_" + i[1] for i in temp.columns]
    temp.sort_values(by=['付款时间_min'], ascending=True, inplace=True)
    temp['time'] = temp['付款时间_min'].apply(lambda x: pd.to_datetime(x).strftime('%Y-%m-%d'))
    temp.rename(columns={'数量_sum': 'data'}, inplace=True)

    # save as excel
    writer = pd.ExcelWriter(output_dir + '\\' + "jiadian_time_series_data.xlsx")
    temp[['data', 'time']].to_excel(writer, sheet_name='day', index=False)

    # week
    temp1 = pd.DataFrame(df.groupby('week').agg({'数量': {'sum'}, '付款时间': {'min'}}))
    # 重命名列名
    temp1.columns = [i[0] + "_" + i[1] for i in temp1.columns]
    temp1.sort_values(by=['付款时间_min'], ascending=True, inplace=True)
    temp1['time'] = temp1['付款时间_min'].apply(lambda x: pd.to_datetime(x).strftime('%Y-%m-%d'))
    temp1.rename(columns={'数量_sum': 'data'}, inplace=True)

    # save as excel
    temp1[['data', 'time']].to_excel(writer, sheet_name='week', index=False)

    # 销量top2的商品大类的day日销量序列（没有销量的日期，销量补充为0）
    df['商品类别'] = df['商品名称'].apply(classify_jiadian)
    temp2 = pd.DataFrame(df[df['商品类别'] == '耳机'].groupby('date').agg({'数量': {'sum'}, '付款时间': {'min'}}))
    temp2.columns = [i[0] + "_" + i[1] for i in temp2.columns]
    temp2.sort_values(by=['付款时间_min'],ascending=True,inplace=True)

    temp3 = pd.DataFrame(df[df['商品类别'] == '平板'].groupby('date').agg({'数量': {'sum'}, '付款时间': {'min'}}))
    temp3.columns = [i[0] + "_" + i[1] for i in temp3.columns]
    temp3.sort_values(by=['付款时间_min'],ascending=True,inplace=True)

    temp11 = pd.merge(temp, temp2, how='left', left_index=True, right_index=True).fillna(0)
    temp22 = pd.merge(temp, temp3, how='left', left_index=True, right_index=True).fillna(0)

    # save as excel
    output = temp11[['数量_sum','time']]
    output.rename(columns={'数量_sum': 'data'}, inplace=True)
    output.to_excel(writer,sheet_name='erji',index=False)

    output = temp22[['数量_sum','time']]
    output.rename(columns={'数量_sum': 'data'}, inplace=True)
    output.to_excel(writer,sheet_name='pingban',index=False)

    writer._save()
    writer.close()