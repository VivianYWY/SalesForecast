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

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# load data

output_dir = '*****\\result\\jiadian'

print(len(ori_df.index.values)) # records num
ori_df.head() # show head data
df = ori_df[['商品名称','数量','实际单价','付款时间']].dropna(axis=0,how='any') # choose fields and drop null
df.drop_duplicates()
print(len(df.index.values)) # records num after filtering

df['销售额'] = df['数量'] * df['实际单价']

# plot
temp1 = pd.DataFrame({'商品名称':df.groupby('商品名称')['数量'].sum().index, '销量':df.groupby('商品名称')['数量'].sum().values})
temp2 = pd.DataFrame({'商品名称':df.groupby('商品名称')['销售额'].sum().index, '销售额':df.groupby('商品名称')['销售额'].sum().values})
plt.figure(figsize=(50, 5))
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus']=False
plt.grid(linestyle='-.')
plt.grid(True)
plt.bar(temp1['商品名称'], temp1['销量'], label="销量",color='blue')
plt.tick_params(axis='x',labelrotation=90)
plt.title('商品细类的销量')
plt.xlabel('商品名称')
plt.ylabel('销量')
plt.savefig(output_dir + '\\' + '商品细类的销量.png', bbox_inches='tight')
plt.show()

plt.clf()
plt.grid(linestyle='-.')
plt.grid(True)
plt.figure(figsize=(50, 5))
plt.bar(temp2['商品名称'], temp2['销售额'], label="销售额",color='orange')
plt.tick_params(axis='x',labelrotation=90)
plt.title('商品细类的销售额')
plt.xlabel('商品名称')
plt.ylabel('销售额')
plt.savefig(output_dir + '\\' + '商品细类的销售额.png', bbox_inches='tight')
plt.show()

# save as excel
# writer = pd.ExcelWriter("dianshang_process.xlsx")
# temp1.to_excel(writer,sheet_name='商品名称_数量')
# writer._save()
# writer.close()

df['商品类别'] = df['商品名称'].apply(classify)
temp1 = pd.DataFrame({'商品类别':df.groupby('商品类别')['数量'].sum().index, '销量':df.groupby('商品类别')['数量'].sum().values})
temp2 = pd.DataFrame({'商品类别':df.groupby('商品类别')['销售额'].sum().index, '销售额':df.groupby('商品类别')['销售额'].sum().values})

plt.clf()
plt.grid(linestyle='-.')
plt.grid(True)
bar = plt.barh(temp1['商品类别'], temp1['销量'],label="销量",color='blue')
plt.bar_label(bar, label_type='edge')
plt.title('商品大类的销量')
plt.xlabel('销量')
plt.ylabel('商品大类')
plt.savefig(output_dir + '\\' + '商品大类的销量.png', bbox_inches='tight')
plt.show()

plt.clf()
plt.grid(linestyle='-.')
plt.grid(True)
bar = plt.barh(temp2['商品类别'], temp2['销售额'], label="销售额",color='orange')
plt.bar_label(bar, label_type='edge')
plt.title('商品大类的销售额')
plt.xlabel('销售额')
plt.ylabel('商品大类')
plt.savefig(output_dir + '\\' + '商品大类的销售额.png', bbox_inches='tight')
plt.show()

df['date'] = pd.to_datetime(df['付款时间']).dt.day.apply(str).str.cat(pd.to_datetime(df['付款时间']).dt.month.apply(str),sep=',').str.cat(pd.to_datetime(df['付款时间']).dt.year.apply(str), sep=',')
df['week'] = pd.to_datetime(df['付款时间']).dt.isocalendar().week.apply(str).str.cat(pd.to_datetime(df['付款时间']).dt.year.apply(str),sep=',')
df['month'] = pd.to_datetime(df['付款时间']).dt.month.apply(str).str.cat(pd.to_datetime(df['付款时间']).dt.year.apply(str),sep=',')

# 日均销量占比80%的TOP N商品细类，第一次卖出该商品细类的时间节点
temp = pd.DataFrame(df.groupby('商品名称').agg({'数量':{'sum'},'付款时间':{'min'},'date':{'nunique'}}))
temp['日均销量'] = (temp['数量']['sum'] / (temp['date']['nunique']))
temp.sort_values(by=['日均销量'],ascending=False,inplace=True)
# Computing cumulative Percentage
temp['cum_percent'] = (temp['日均销量'].cumsum() / temp['日均销量'].sum()) * 100
temp = temp[temp['cum_percent'] < 80]
print(len(temp.index.values)) # records num after filtering

temp['timebase'] = '2024-6-1'
temp['days'] =(pd.to_datetime(temp['timebase']) - pd.to_datetime(temp['付款时间']['min'])).dt.days
temp.sort_values(by=['days'],ascending=False,inplace=True)

plt.clf()
plt.figure(figsize=(50, 5))
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus']=False
plt.grid(linestyle='-.')
plt.grid(True)
plt.bar(temp.index, temp['days'], label="天数",color='red')
plt.tick_params(axis='x',labelrotation=90)
plt.title('商品细类第一次出现至今的天数')
plt.xlabel('商品名称')
plt.ylabel('天数')
plt.savefig(output_dir + '\\' + '商品细类第一次出现至今的天数.png', bbox_inches='tight')
plt.show()

plt.clf()
plt.figure(figsize=(50, 5))
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus']=False
plt.grid(linestyle='-.')
plt.grid(True)
plt.bar(temp.index, temp['日均销量'], label="日均销量",color='green')
plt.tick_params(axis='x',labelrotation=90)
plt.title('商品细类日均销量（占比前80%）')
plt.xlabel('商品名称')
plt.ylabel('日均销量')
plt.savefig(output_dir + '\\' + '商品细类日均销量（占比前80%）.png', bbox_inches='tight')
plt.show()

# 有销量的天数（平均）
# 重命名列名
temp.columns = [i[0] + "_" + i[1] for i in temp.columns]
print(temp['date_nunique'].mean())
print(temp.iloc[0:31,[2]].mean()) # 日均销量top30的有销量的天数（平均）

# # date-num
temp = pd.DataFrame(df.groupby('date').agg({'数量':{'sum'},'付款时间':{'max'}}))
temp.sort_values(by=[('付款时间', 'max')], inplace=True)
temp1 = pd.DataFrame(df[df['商品类别']=='耳机'].groupby('date').agg({'数量':{'sum'},'付款时间':{'max'}}))
temp1.sort_values(by=[('付款时间', 'max')], inplace=True)
temp2 = pd.DataFrame(df[df['商品类别']=='屏幕'].groupby('date').agg({'数量':{'sum'},'付款时间':{'max'}}))
temp2.sort_values(by=[('付款时间', 'max')], inplace=True)
temp3 = pd.DataFrame(df[df['商品类别']=='平板'].groupby('date').agg({'数量':{'sum'},'付款时间':{'max'}}))
temp3.sort_values(by=[('付款时间', 'max')], inplace=True)
temp4 = pd.DataFrame(df[df['商品类别']=='电视机'].groupby('date').agg({'数量':{'sum'},'付款时间':{'max'}}))
temp4.sort_values(by=[('付款时间', 'max')], inplace=True)
temp5 = pd.DataFrame(df[df['商品类别']=='门锁'].groupby('date').agg({'数量':{'sum'},'付款时间':{'max'}}))
temp5.sort_values(by=[('付款时间', 'max')], inplace=True)
temp6 = pd.DataFrame(df[df['商品类别']=='电脑'].groupby('date').agg({'数量':{'sum'},'付款时间':{'max'}}))
temp6.sort_values(by=[('付款时间', 'max')], inplace=True)
temp7 = pd.DataFrame(df[df['商品类别']=='其他'].groupby('date').agg({'数量':{'sum'},'付款时间':{'max'}}))
temp7.sort_values(by=[('付款时间', 'max')], inplace=True)

temp11 = pd.merge(temp,temp1,how='left',left_index=True, right_index=True).fillna(0)
temp22 = pd.merge(temp,temp2,how='left',left_index=True, right_index=True).fillna(0)
temp33 = pd.merge(temp,temp3,how='left',left_index=True, right_index=True).fillna(0)
temp44 = pd.merge(temp,temp4,how='left',left_index=True, right_index=True).fillna(0)
temp55 = pd.merge(temp,temp5,how='left',left_index=True, right_index=True).fillna(0)
temp66 = pd.merge(temp,temp6,how='left',left_index=True, right_index=True).fillna(0)
temp77 = pd.merge(temp,temp7,how='left',left_index=True, right_index=True).fillna(0)

plt.clf()
plt.figure(figsize=(50, 5))
plt.plot(temp.index, temp['数量']['sum'], c='red', label="总销量",marker=".")
plt.plot(temp11.index, temp11['数量_y']['sum'], c='gray', label="耳机",marker="s")
plt.plot(temp22.index, temp22['数量_y']['sum'], c='green', label="屏幕",marker="*")
plt.plot(temp33.index, temp33['数量_y']['sum'], c='blue', label="平板",marker="o")
plt.plot(temp44.index, temp44['数量_y']['sum'], c='olive', label="电视机",marker="^")
plt.plot(temp55.index, temp55['数量_y']['sum'], c='purple', label="门锁",marker="p")
plt.plot(temp66.index, temp66['数量_y']['sum'], c='pink', label="电脑",marker="X")
plt.plot(temp77.index, temp77['数量_y']['sum'], c='brown', label="其他",marker="+")
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tick_params(axis='x',labelrotation=90)
plt.xlabel("日期", fontdict={'size': 16})
plt.ylabel("销量", fontdict={'size': 16})
plt.title("日销量变化", fontdict={'size': 20})
plt.savefig(output_dir + '\\' + '日销量变化.png', bbox_inches='tight')
plt.show()

# # week-num
temp = pd.DataFrame(df.groupby('week').agg({'数量':{'sum'},'付款时间':{'max'}}))
temp.sort_values(by=[('付款时间', 'max')], inplace=True)
temp1 = pd.DataFrame(df[df['商品类别']=='耳机'].groupby('week').agg({'数量':{'sum'},'付款时间':{'max'}}))
temp1.sort_values(by=[('付款时间', 'max')], inplace=True)
temp2 = pd.DataFrame(df[df['商品类别']=='屏幕'].groupby('week').agg({'数量':{'sum'},'付款时间':{'max'}}))
temp2.sort_values(by=[('付款时间', 'max')], inplace=True)
temp3 = pd.DataFrame(df[df['商品类别']=='平板'].groupby('week').agg({'数量':{'sum'},'付款时间':{'max'}}))
temp3.sort_values(by=[('付款时间', 'max')], inplace=True)
temp4 = pd.DataFrame(df[df['商品类别']=='电视机'].groupby('week').agg({'数量':{'sum'},'付款时间':{'max'}}))
temp4.sort_values(by=[('付款时间', 'max')], inplace=True)
temp5 = pd.DataFrame(df[df['商品类别']=='门锁'].groupby('week').agg({'数量':{'sum'},'付款时间':{'max'}}))
temp5.sort_values(by=[('付款时间', 'max')], inplace=True)
temp6 = pd.DataFrame(df[df['商品类别']=='电脑'].groupby('week').agg({'数量':{'sum'},'付款时间':{'max'}}))
temp6.sort_values(by=[('付款时间', 'max')], inplace=True)
temp7 = pd.DataFrame(df[df['商品类别']=='其他'].groupby('week').agg({'数量':{'sum'},'付款时间':{'max'}}))
temp7.sort_values(by=[('付款时间', 'max')], inplace=True)

temp11 = pd.merge(temp,temp1,how='left',left_index=True, right_index=True).fillna(0)
temp22 = pd.merge(temp,temp2,how='left',left_index=True, right_index=True).fillna(0)
temp33 = pd.merge(temp,temp3,how='left',left_index=True, right_index=True).fillna(0)
temp44 = pd.merge(temp,temp4,how='left',left_index=True, right_index=True).fillna(0)
temp55 = pd.merge(temp,temp5,how='left',left_index=True, right_index=True).fillna(0)
temp66 = pd.merge(temp,temp6,how='left',left_index=True, right_index=True).fillna(0)
temp77 = pd.merge(temp,temp7,how='left',left_index=True, right_index=True).fillna(0)

plt.clf()
plt.figure(figsize=(50, 5))
plt.plot(temp.index, temp['数量']['sum'], c='red', label="总销量",marker=".")
plt.plot(temp11.index, temp11['数量_y']['sum'], c='gray', label="耳机",marker="s")
plt.plot(temp22.index, temp22['数量_y']['sum'], c='green', label="屏幕",marker="*")
plt.plot(temp33.index, temp33['数量_y']['sum'], c='blue', label="平板",marker="o")
plt.plot(temp44.index, temp44['数量_y']['sum'], c='olive', label="电视机",marker="^")
plt.plot(temp55.index, temp55['数量_y']['sum'], c='purple', label="门锁",marker="p")
plt.plot(temp66.index, temp66['数量_y']['sum'], c='pink', label="电脑",marker="X")
plt.plot(temp77.index, temp77['数量_y']['sum'], c='brown', label="其他",marker="+")
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tick_params(axis='x',labelrotation=90)
plt.xlabel("周数", fontdict={'size': 16})
plt.ylabel("销量", fontdict={'size': 16})
plt.title("周销量变化", fontdict={'size': 20})
plt.savefig(output_dir + '\\' + '周销量变化.png', bbox_inches='tight')
plt.show()

# # month-num
temp = pd.DataFrame(df.groupby('month').agg({'数量':{'sum'},'付款时间':{'max'}}))
temp.sort_values(by=[('付款时间', 'max')], inplace=True)
temp1 = pd.DataFrame(df[df['商品类别']=='耳机'].groupby('month').agg({'数量':{'sum'},'付款时间':{'max'}}))
temp1.sort_values(by=[('付款时间', 'max')], inplace=True)
temp2 = pd.DataFrame(df[df['商品类别']=='屏幕'].groupby('month').agg({'数量':{'sum'},'付款时间':{'max'}}))
temp2.sort_values(by=[('付款时间', 'max')], inplace=True)
temp3 = pd.DataFrame(df[df['商品类别']=='平板'].groupby('month').agg({'数量':{'sum'},'付款时间':{'max'}}))
temp3.sort_values(by=[('付款时间', 'max')], inplace=True)
temp4 = pd.DataFrame(df[df['商品类别']=='电视机'].groupby('month').agg({'数量':{'sum'},'付款时间':{'max'}}))
temp4.sort_values(by=[('付款时间', 'max')], inplace=True)
temp5 = pd.DataFrame(df[df['商品类别']=='门锁'].groupby('month').agg({'数量':{'sum'},'付款时间':{'max'}}))
temp5.sort_values(by=[('付款时间', 'max')], inplace=True)
temp6 = pd.DataFrame(df[df['商品类别']=='电脑'].groupby('month').agg({'数量':{'sum'},'付款时间':{'max'}}))
temp6.sort_values(by=[('付款时间', 'max')], inplace=True)
temp7 = pd.DataFrame(df[df['商品类别']=='其他'].groupby('month').agg({'数量':{'sum'},'付款时间':{'max'}}))
temp7.sort_values(by=[('付款时间', 'max')], inplace=True)

temp11 = pd.merge(temp,temp1,how='left',left_index=True, right_index=True).fillna(0)
temp22 = pd.merge(temp,temp2,how='left',left_index=True, right_index=True).fillna(0)
temp33 = pd.merge(temp,temp3,how='left',left_index=True, right_index=True).fillna(0)
temp44 = pd.merge(temp,temp4,how='left',left_index=True, right_index=True).fillna(0)
temp55 = pd.merge(temp,temp5,how='left',left_index=True, right_index=True).fillna(0)
temp66 = pd.merge(temp,temp6,how='left',left_index=True, right_index=True).fillna(0)
temp77 = pd.merge(temp,temp7,how='left',left_index=True, right_index=True).fillna(0)

plt.clf()
plt.plot(temp.index, temp['数量']['sum'], c='red', label="总销量",marker=".")
plt.plot(temp11.index, temp11['数量_y']['sum'], c='gray', label="耳机",marker="s")
plt.plot(temp22.index, temp22['数量_y']['sum'], c='green', label="屏幕",marker="*")
plt.plot(temp33.index, temp33['数量_y']['sum'], c='blue', label="平板",marker="o")
plt.plot(temp44.index, temp44['数量_y']['sum'], c='olive', label="电视机",marker="^")
plt.plot(temp55.index, temp55['数量_y']['sum'], c='purple', label="门锁",marker="p")
plt.plot(temp66.index, temp66['数量_y']['sum'], c='pink', label="电脑",marker="X")
plt.plot(temp77.index, temp77['数量_y']['sum'], c='brown', label="其他",marker="+")
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tick_params(axis='x',labelrotation=90)
plt.xlabel("月份", fontdict={'size': 16})
plt.ylabel("销量", fontdict={'size': 16})
plt.title("月销量变化", fontdict={'size': 20})
plt.savefig(output_dir + '\\' + '月销量变化.png', bbox_inches='tight')
plt.show()

# # date-money
temp = pd.DataFrame(df.groupby('date').agg({'销售额':{'sum'},'付款时间':{'max'}}))
temp.sort_values(by=[('付款时间', 'max')], inplace=True)
temp1 = pd.DataFrame(df[df['商品类别']=='耳机'].groupby('date').agg({'销售额':{'sum'},'付款时间':{'max'}}))
temp1.sort_values(by=[('付款时间', 'max')], inplace=True)
temp2 = pd.DataFrame(df[df['商品类别']=='屏幕'].groupby('date').agg({'销售额':{'sum'},'付款时间':{'max'}}))
temp2.sort_values(by=[('付款时间', 'max')], inplace=True)
temp3 = pd.DataFrame(df[df['商品类别']=='平板'].groupby('date').agg({'销售额':{'sum'},'付款时间':{'max'}}))
temp3.sort_values(by=[('付款时间', 'max')], inplace=True)
temp4 = pd.DataFrame(df[df['商品类别']=='电视机'].groupby('date').agg({'销售额':{'sum'},'付款时间':{'max'}}))
temp4.sort_values(by=[('付款时间', 'max')], inplace=True)
temp5 = pd.DataFrame(df[df['商品类别']=='门锁'].groupby('date').agg({'销售额':{'sum'},'付款时间':{'max'}}))
temp5.sort_values(by=[('付款时间', 'max')], inplace=True)
temp6 = pd.DataFrame(df[df['商品类别']=='电脑'].groupby('date').agg({'销售额':{'sum'},'付款时间':{'max'}}))
temp6.sort_values(by=[('付款时间', 'max')], inplace=True)
temp7 = pd.DataFrame(df[df['商品类别']=='其他'].groupby('date').agg({'销售额':{'sum'},'付款时间':{'max'}}))
temp7.sort_values(by=[('付款时间', 'max')], inplace=True)

temp11 = pd.merge(temp,temp1,how='left',left_index=True, right_index=True).fillna(0)
temp22 = pd.merge(temp,temp2,how='left',left_index=True, right_index=True).fillna(0)
temp33 = pd.merge(temp,temp3,how='left',left_index=True, right_index=True).fillna(0)
temp44 = pd.merge(temp,temp4,how='left',left_index=True, right_index=True).fillna(0)
temp55 = pd.merge(temp,temp5,how='left',left_index=True, right_index=True).fillna(0)
temp66 = pd.merge(temp,temp6,how='left',left_index=True, right_index=True).fillna(0)
temp77 = pd.merge(temp,temp7,how='left',left_index=True, right_index=True).fillna(0)

plt.clf()
plt.figure(figsize=(50, 5))
plt.plot(temp.index, temp['销售额']['sum'], c='red', label="总销售额",marker=".")
plt.plot(temp11.index, temp11['销售额_y']['sum'], c='gray', label="耳机",marker="s")
plt.plot(temp22.index, temp22['销售额_y']['sum'], c='green', label="屏幕",marker="*")
plt.plot(temp33.index, temp33['销售额_y']['sum'], c='blue', label="平板",marker="o")
plt.plot(temp44.index, temp44['销售额_y']['sum'], c='olive', label="电视机",marker="^")
plt.plot(temp55.index, temp55['销售额_y']['sum'], c='purple', label="门锁",marker="p")
plt.plot(temp66.index, temp66['销售额_y']['sum'], c='pink', label="电脑",marker="X")
plt.plot(temp77.index, temp77['销售额_y']['sum'], c='brown', label="其他",marker="+")
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tick_params(axis='x',labelrotation=90)
plt.xlabel("日期", fontdict={'size': 16})
plt.ylabel("销售额", fontdict={'size': 16})
plt.title("日销售额变化", fontdict={'size': 20})
plt.savefig(output_dir + '\\' + '日销售额变化.png', bbox_inches='tight')
plt.show()

# # week-money
temp = pd.DataFrame(df.groupby('week').agg({'销售额':{'sum'},'付款时间':{'max'}}))
temp.sort_values(by=[('付款时间', 'max')], inplace=True)
temp1 = pd.DataFrame(df[df['商品类别']=='耳机'].groupby('week').agg({'销售额':{'sum'},'付款时间':{'max'}}))
temp1.sort_values(by=[('付款时间', 'max')], inplace=True)
temp2 = pd.DataFrame(df[df['商品类别']=='屏幕'].groupby('week').agg({'销售额':{'sum'},'付款时间':{'max'}}))
temp2.sort_values(by=[('付款时间', 'max')], inplace=True)
temp3 = pd.DataFrame(df[df['商品类别']=='平板'].groupby('week').agg({'销售额':{'sum'},'付款时间':{'max'}}))
temp3.sort_values(by=[('付款时间', 'max')], inplace=True)
temp4 = pd.DataFrame(df[df['商品类别']=='电视机'].groupby('week').agg({'销售额':{'sum'},'付款时间':{'max'}}))
temp4.sort_values(by=[('付款时间', 'max')], inplace=True)
temp5 = pd.DataFrame(df[df['商品类别']=='门锁'].groupby('week').agg({'销售额':{'sum'},'付款时间':{'max'}}))
temp5.sort_values(by=[('付款时间', 'max')], inplace=True)
temp6 = pd.DataFrame(df[df['商品类别']=='电脑'].groupby('week').agg({'销售额':{'sum'},'付款时间':{'max'}}))
temp6.sort_values(by=[('付款时间', 'max')], inplace=True)
temp7 = pd.DataFrame(df[df['商品类别']=='其他'].groupby('week').agg({'销售额':{'sum'},'付款时间':{'max'}}))
temp7.sort_values(by=[('付款时间', 'max')], inplace=True)

temp11 = pd.merge(temp,temp1,how='left',left_index=True, right_index=True).fillna(0)
temp22 = pd.merge(temp,temp2,how='left',left_index=True, right_index=True).fillna(0)
temp33 = pd.merge(temp,temp3,how='left',left_index=True, right_index=True).fillna(0)
temp44 = pd.merge(temp,temp4,how='left',left_index=True, right_index=True).fillna(0)
temp55 = pd.merge(temp,temp5,how='left',left_index=True, right_index=True).fillna(0)
temp66 = pd.merge(temp,temp6,how='left',left_index=True, right_index=True).fillna(0)
temp77 = pd.merge(temp,temp7,how='left',left_index=True, right_index=True).fillna(0)

plt.clf()
plt.figure(figsize=(50, 5))
plt.plot(temp.index, temp['销售额']['sum'], c='red', label="总销售额",marker=".")
plt.plot(temp11.index, temp11['销售额_y']['sum'], c='gray', label="耳机",marker="s")
plt.plot(temp22.index, temp22['销售额_y']['sum'], c='green', label="屏幕",marker="*")
plt.plot(temp33.index, temp33['销售额_y']['sum'], c='blue', label="平板",marker="o")
plt.plot(temp44.index, temp44['销售额_y']['sum'], c='olive', label="电视机",marker="^")
plt.plot(temp55.index, temp55['销售额_y']['sum'], c='purple', label="门锁",marker="p")
plt.plot(temp66.index, temp66['销售额_y']['sum'], c='pink', label="电脑",marker="X")
plt.plot(temp77.index, temp77['销售额_y']['sum'], c='brown', label="其他",marker="+")
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tick_params(axis='x',labelrotation=90)
plt.xlabel("周数", fontdict={'size': 16})
plt.ylabel("销售额", fontdict={'size': 16})
plt.title("周销售额变化", fontdict={'size': 20})
plt.savefig(output_dir + '\\' + '周销售额变化.png', bbox_inches='tight')
plt.show()

# # month-money
temp = pd.DataFrame(df.groupby('month').agg({'销售额':{'sum'},'付款时间':{'max'}}))
temp.sort_values(by=[('付款时间', 'max')], inplace=True)
temp1 = pd.DataFrame(df[df['商品类别']=='耳机'].groupby('month').agg({'销售额':{'sum'},'付款时间':{'max'}}))
temp1.sort_values(by=[('付款时间', 'max')], inplace=True)
temp2 = pd.DataFrame(df[df['商品类别']=='屏幕'].groupby('month').agg({'销售额':{'sum'},'付款时间':{'max'}}))
temp2.sort_values(by=[('付款时间', 'max')], inplace=True)
temp3 = pd.DataFrame(df[df['商品类别']=='平板'].groupby('month').agg({'销售额':{'sum'},'付款时间':{'max'}}))
temp3.sort_values(by=[('付款时间', 'max')], inplace=True)
temp4 = pd.DataFrame(df[df['商品类别']=='电视机'].groupby('month').agg({'销售额':{'sum'},'付款时间':{'max'}}))
temp4.sort_values(by=[('付款时间', 'max')], inplace=True)
temp5 = pd.DataFrame(df[df['商品类别']=='门锁'].groupby('month').agg({'销售额':{'sum'},'付款时间':{'max'}}))
temp5.sort_values(by=[('付款时间', 'max')], inplace=True)
temp6 = pd.DataFrame(df[df['商品类别']=='电脑'].groupby('month').agg({'销售额':{'sum'},'付款时间':{'max'}}))
temp6.sort_values(by=[('付款时间', 'max')], inplace=True)
temp7 = pd.DataFrame(df[df['商品类别']=='其他'].groupby('month').agg({'销售额':{'sum'},'付款时间':{'max'}}))
temp7.sort_values(by=[('付款时间', 'max')], inplace=True)

temp11 = pd.merge(temp,temp1,how='left',left_index=True, right_index=True).fillna(0)
temp22 = pd.merge(temp,temp2,how='left',left_index=True, right_index=True).fillna(0)
temp33 = pd.merge(temp,temp3,how='left',left_index=True, right_index=True).fillna(0)
temp44 = pd.merge(temp,temp4,how='left',left_index=True, right_index=True).fillna(0)
temp55 = pd.merge(temp,temp5,how='left',left_index=True, right_index=True).fillna(0)
temp66 = pd.merge(temp,temp6,how='left',left_index=True, right_index=True).fillna(0)
temp77 = pd.merge(temp,temp7,how='left',left_index=True, right_index=True).fillna(0)

plt.clf()
plt.plot(temp.index, temp['销售额']['sum'], c='red', label="总销售额",marker=".")
plt.plot(temp11.index, temp11['销售额_y']['sum'], c='gray', label="耳机",marker="s")
plt.plot(temp22.index, temp22['销售额_y']['sum'], c='green', label="屏幕",marker="*")
plt.plot(temp33.index, temp33['销售额_y']['sum'], c='blue', label="平板",marker="o")
plt.plot(temp44.index, temp44['销售额_y']['sum'], c='olive', label="电视机",marker="^")
plt.plot(temp55.index, temp55['销售额_y']['sum'], c='purple', label="门锁",marker="p")
plt.plot(temp66.index, temp66['销售额_y']['sum'], c='pink', label="电脑",marker="X")
plt.plot(temp77.index, temp77['销售额_y']['sum'], c='brown', label="其他",marker="+")
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tick_params(axis='x',labelrotation=90)
plt.xlabel("月份", fontdict={'size': 16})
plt.ylabel("销售额", fontdict={'size': 16})
plt.title("月销售额变化", fontdict={'size': 20})
plt.savefig(output_dir + '\\' + '月销售额变化.png', bbox_inches='tight')
plt.show()

# # date-item class
temp = pd.DataFrame(df.groupby('date').agg({'商品名称':{'nunique'},'付款时间':{'max'}}))
temp.sort_values(by=[('付款时间', 'max')], inplace=True)
temp1 = pd.DataFrame(df.groupby('date').agg({'商品类别':{'nunique'},'付款时间':{'max'}}))
temp1.sort_values(by=[('付款时间', 'max')], inplace=True)

temp11 = pd.merge(temp,temp1,how='left',left_index=True, right_index=True).fillna(0)

plt.clf()
plt.figure(figsize=(50, 5))
plt.plot(temp.index, temp['商品名称']['nunique'], c='red', label="商品细类",marker=".")
plt.plot(temp11.index, temp11['商品类别']['nunique'], c='blue', label="商品大类",marker="s")
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tick_params(axis='x',labelrotation=90)
plt.xlabel("日期", fontdict={'size': 16})
plt.ylabel("商品类别个数", fontdict={'size': 16})
plt.title("商品类别日变化", fontdict={'size': 20})
plt.savefig(output_dir + '\\' + '商品类别日变化.png', bbox_inches='tight')
plt.show()

# # week-item class
temp = pd.DataFrame(df.groupby('week').agg({'商品名称':{'nunique'},'付款时间':{'max'}}))
temp.sort_values(by=[('付款时间', 'max')], inplace=True)
temp1 = pd.DataFrame(df.groupby('week').agg({'商品类别':{'nunique'},'付款时间':{'max'}}))
temp1.sort_values(by=[('付款时间', 'max')], inplace=True)

temp11 = pd.merge(temp,temp1,how='left',left_index=True, right_index=True).fillna(0)

plt.clf()
plt.figure(figsize=(50, 5))
plt.plot(temp.index, temp['商品名称']['nunique'], c='red', label="商品细类",marker=".")
plt.plot(temp11.index, temp11['商品类别']['nunique'], c='blue', label="商品大类",marker="s")
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tick_params(axis='x',labelrotation=90)
plt.xlabel("周数", fontdict={'size': 16})
plt.ylabel("商品类别个数", fontdict={'size': 16})
plt.title("商品类别周变化", fontdict={'size': 20})
plt.savefig(output_dir + '\\' + '商品类别周变化.png', bbox_inches='tight')
plt.show()

# # month-item class
temp = pd.DataFrame(df.groupby('month').agg({'商品名称':{'nunique'},'付款时间':{'max'}}))
temp.sort_values(by=[('付款时间', 'max')], inplace=True)
temp1 = pd.DataFrame(df.groupby('month').agg({'商品类别':{'nunique'},'付款时间':{'max'}}))
temp1.sort_values(by=[('付款时间', 'max')], inplace=True)

temp11 = pd.merge(temp,temp1,how='left',left_index=True, right_index=True).fillna(0)

plt.clf()
plt.plot(temp.index, temp['商品名称']['nunique'], c='red', label="商品细类",marker=".")
plt.plot(temp11.index, temp11['商品类别']['nunique'], c='blue', label="商品大类",marker="s")
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tick_params(axis='x',labelrotation=90)
plt.xlabel("月份", fontdict={'size': 16})
plt.ylabel("商品类别个数", fontdict={'size': 16})
plt.title("商品类别月变化", fontdict={'size': 20})
plt.savefig(output_dir + '\\' + '商品类别月变化.png', bbox_inches='tight')
plt.show()

def decimal_process(x):
    x = round(x,4)
    return x

# day label
DP = DateProcess(df['付款时间'])
df_day_label = DP.get_Feature_day()
new_df = pd.concat([df,df_day_label[['quarter','day_type_chinese','sales_type_exact','sales_type_range','school_leave']]],axis=1)

# 假期
new_df['flag']=new_df.apply(lambda x:'非假期' if x.day_type_chinese == '工作日' or x.day_type_chinese == '周末或调休' else '假期',axis=1)
temp = pd.DataFrame(new_df.groupby('flag').agg({'数量':{'sum'},'销售额':{'sum'},'date':{'nunique'}}))
temp['日均销量'] = (temp['数量']['sum'] / (temp['date']['nunique']))
temp['日均销售额'] = temp['销售额']['sum'] / (temp['date']['nunique'])
temp1 = pd.DataFrame(new_df[(new_df['day_type_chinese'] != '工作日') & (new_df['day_type_chinese'] != '周末或调休')].groupby('day_type_chinese').agg({'数量':{'sum'},'销售额':{'sum'},'date':{'nunique'}}))
temp1['日均销量'] = temp1['数量']['sum'] / (temp1['date']['nunique'])
temp1['日均销售额'] = temp1['销售额']['sum'] / (temp1['date']['nunique'])
temp_all = pd.concat([temp[['日均销量','日均销售额']],temp1[['日均销量','日均销售额']]])
#TODO: 取小数位仍然有问题，待fix
#temp_all['日均销量'] = pd.to_numeric(temp_all['日均销量']).map("{:.4f}".format)

plt.clf()
plt.grid(linestyle='-.')
plt.grid(True)
bar = plt.bar(temp_all.index, temp_all['日均销量'], label="日均销量",color='blue')
plt.bar_label(bar, label_type='edge')
plt.title('日均销量（假期）')
plt.ylabel('日均销量')
plt.savefig(output_dir + '\\' + '日均销量（假期）.png', bbox_inches='tight')
plt.show()

plt.clf()
plt.grid(linestyle='-.')
plt.grid(True)
bar = plt.bar(temp_all.index, temp_all['日均销售额'], label="日均销售额",color='orange')
plt.bar_label(bar, label_type='edge')
plt.title('日均销售额（假期）')
plt.ylabel('日均销售额')
plt.savefig(output_dir + '\\' + '日均销售额（假期）.png', bbox_inches='tight')
plt.show()

# plt.rcParams['font.sans-serif'] = ['SimHei']
# fig, ax = plt.subplots()
# x = np.arange(len(temp_all.index.values))
# width = 0.35  # the width of the bars
# rects1 = ax.bar(x - width / 2, temp_all['日均销量'], width, label='日均销量')
# rects2 = ax.bar(x + width / 2, temp_all['日均销售额'], width, label='日均销售额')
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('日均销量和销售额')
# ax.set_title('日均销量和销售额（假期）')
# ax.set_xticks(x)
# ax.set_xticklabels(temp_all.index)
# ax.legend()
# autolabel(rects1)
# autolabel(rects2)
# fig.tight_layout()
# plt.savefig(output_dir + '\\' + '日均销量和销售额（假期）.png', bbox_inches='tight')
# plt.show()

# 工作日
new_df['flag']=new_df.apply(lambda x:'工作日' if x.day_type_chinese == '工作日' else '非工作日',axis=1)
temp = pd.DataFrame(new_df.groupby('flag').agg({'数量':{'sum'},'销售额':{'sum'},'date':{'nunique'}}))
temp['日均销量'] = temp['数量']['sum'] / (temp['date']['nunique'])
temp['日均销售额'] = temp['销售额']['sum'] / (temp['date']['nunique'])

plt.clf()
plt.grid(linestyle='-.')
plt.grid(True)
bar = plt.bar(temp.index, temp['日均销量'], label="日均销量",color='blue')
plt.bar_label(bar, label_type='edge')
plt.title('日均销量（工作日）')
plt.ylabel('日均销量')
plt.savefig(output_dir + '\\' + '日均销量（工作日）.png', bbox_inches='tight')
plt.show()

plt.clf()
plt.grid(linestyle='-.')
plt.grid(True)
bar = plt.bar(temp.index, temp['日均销售额'], label="日均销售额",color='orange')
plt.bar_label(bar, label_type='edge')
plt.title('日均销售额（工作日）')
plt.ylabel('日均销售额')
plt.savefig(output_dir + '\\' + '日均销售额（工作日）.png', bbox_inches='tight')
plt.show()

# 季度
temp = pd.DataFrame(new_df.groupby('quarter').agg({'数量':{'sum'},'销售额':{'sum'},'date':{'nunique'}}))
temp['日均销量'] = temp['数量']['sum'] / (temp['date']['nunique'])
temp['日均销售额'] = temp['销售额']['sum'] / (temp['date']['nunique'])

plt.clf()
plt.grid(linestyle='-.')
plt.grid(True)
bar = plt.bar(temp.index, temp['日均销量'], label="日均销量",color='blue')
plt.bar_label(bar, label_type='edge')
plt.title('日均销量（季度）')
plt.ylabel('日均销量')
plt.savefig(output_dir + '\\' + '日均销量（季度）.png', bbox_inches='tight')
plt.show()

plt.clf()
plt.grid(linestyle='-.')
plt.grid(True)
bar = plt.bar(temp.index, temp['日均销售额'], label="日均销售额",color='orange')
plt.bar_label(bar, label_type='edge')
plt.title('日均销售额（季度）')
plt.ylabel('日均销售额')
plt.savefig(output_dir + '\\' + '日均销售额（季度）.png', bbox_inches='tight')
plt.show()

# 寒暑假
new_df['flag']=new_df.apply(lambda x:'寒暑假' if x.school_leave == '寒假' or x.school_leave == '暑假' else '非寒暑假',axis=1)
temp = pd.DataFrame(new_df.groupby('flag').agg({'数量':{'sum'},'销售额':{'sum'},'date':{'nunique'}}))
temp['日均销量'] = temp['数量']['sum'] / (temp['date']['nunique'])
temp['日均销售额'] = temp['销售额']['sum'] / (temp['date']['nunique'])
temp1 = pd.DataFrame(new_df[new_df['school_leave'] != ''].groupby('school_leave').agg({'数量':{'sum'},'销售额':{'sum'},'date':{'nunique'}}))
temp1['日均销量'] = temp1['数量']['sum'] / (temp1['date']['nunique'])
temp1['日均销售额'] = temp1['销售额']['sum'] / (temp1['date']['nunique'])
temp_all = pd.concat([temp[['日均销量','日均销售额']],temp1[['日均销量','日均销售额']]])

plt.clf()
plt.grid(linestyle='-.')
plt.grid(True)
bar = plt.bar(temp_all.index, temp_all['日均销量'], label="日均销量",color='blue')
plt.bar_label(bar, label_type='edge')
plt.title('日均销量（寒暑假）')
plt.ylabel('日均销量')
plt.savefig(output_dir + '\\' + '日均销量（寒暑假）.png', bbox_inches='tight')
plt.show()

plt.clf()
plt.grid(linestyle='-.')
plt.grid(True)
bar = plt.bar(temp_all.index, temp_all['日均销售额'], label="日均销售额",color='orange')
plt.bar_label(bar, label_type='edge')
plt.title('日均销售额（寒暑假）')
plt.ylabel('日均销售额')
plt.savefig(output_dir + '\\' + '日均销售额（寒暑假）.png', bbox_inches='tight')
plt.show()

#  促销-精确
new_df['flag']=new_df.apply(lambda x:'促销期' if x.sales_type_exact != '' else '非促销期',axis=1)
temp = pd.DataFrame(new_df.groupby('flag').agg({'数量':{'sum'},'销售额':{'sum'},'date':{'nunique'}}))
temp['日均销量'] = temp['数量']['sum'] / (temp['date']['nunique'])
temp['日均销售额'] = temp['销售额']['sum'] / (temp['date']['nunique'])
temp1 = pd.DataFrame(new_df[new_df['sales_type_exact'] != ''].groupby('sales_type_exact').agg({'数量':{'sum'},'销售额':{'sum'},'date':{'nunique'}}))
temp1['日均销量'] = temp1['数量']['sum'] / (temp1['date']['nunique'])
temp1['日均销售额'] = temp1['销售额']['sum'] / (temp1['date']['nunique'])
temp_all = pd.concat([temp[['日均销量','日均销售额']],temp1[['日均销量','日均销售额']]])

plt.clf()
plt.grid(linestyle='-.')
plt.grid(True)
bar = plt.bar(temp_all.index, temp_all['日均销量'], label="日均销量",color='blue')
plt.bar_label(bar, label_type='edge')
plt.title('日均销量（促销-精确）')
plt.ylabel('日均销量')
plt.savefig(output_dir + '\\' + '日均销量（促销-精确）.png', bbox_inches='tight')
plt.show()

plt.clf()
plt.grid(linestyle='-.')
plt.grid(True)
bar = plt.bar(temp_all.index, temp_all['日均销售额'], label="日均销售额",color='orange')
plt.bar_label(bar, label_type='edge')
plt.title('日均销售额（促销-精确）')
plt.ylabel('日均销售额')
plt.savefig(output_dir + '\\' + '日均销售额（促销-精确）.png', bbox_inches='tight')
plt.show()

#  促销-范围
new_df['flag']=new_df.apply(lambda x:'促销期' if x.sales_type_range != '' else '非促销期',axis=1)
temp = pd.DataFrame(new_df.groupby('flag').agg({'数量':{'sum'},'销售额':{'sum'},'date':{'nunique'}}))
temp['日均销量'] = temp['数量']['sum'] / (temp['date']['nunique'])
temp['日均销售额'] = temp['销售额']['sum'] / (temp['date']['nunique'])
temp1 = pd.DataFrame(new_df[new_df['sales_type_range'] != ''].groupby('sales_type_range').agg({'数量':{'sum'},'销售额':{'sum'},'date':{'nunique'}}))
temp1['日均销量'] = temp1['数量']['sum'] / (temp1['date']['nunique'])
temp1['日均销售额'] = temp1['销售额']['sum'] / (temp1['date']['nunique'])
temp_all = pd.concat([temp[['日均销量','日均销售额']],temp1[['日均销量','日均销售额']]])

plt.clf()
plt.grid(linestyle='-.')
plt.grid(True)
bar = plt.bar(temp_all.index, temp_all['日均销量'], label="日均销量",color='blue')
plt.bar_label(bar, label_type='edge')
plt.title('日均销量（促销-范围）')
plt.ylabel('日均销量')
plt.savefig(output_dir + '\\' + '日均销量（促销-范围）.png', bbox_inches='tight')
plt.show()

plt.clf()
plt.grid(linestyle='-.')
plt.grid(True)
bar = plt.bar(temp_all.index, temp_all['日均销售额'], label="日均销售额",color='orange')
plt.bar_label(bar, label_type='edge')
plt.title('日均销售额（促销-范围）')
plt.ylabel('日均销售额')
plt.savefig(output_dir + '\\' + '日均销售额（促销-范围）.png', bbox_inches='tight')
plt.show()

# 价格和销量
temp = pd.DataFrame(df[df['实际单价']!=0].groupby('实际单价')['数量'].sum())
temp['实际单价'] = temp.index
df.plot.scatter(x='实际单价', y='数量')
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus']=False
plt.grid(linestyle='-.')
plt.grid(True)
width = 0.1
plt.title('价格与销量')
plt.xlabel('商品单价')
plt.ylabel('商品销量')
plt.savefig(output_dir + '\\' + '价格与销量.png', bbox_inches='tight')
plt.show()











