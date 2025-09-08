import os
import time
import datetime
from datetime import timedelta
import pandas as pd
from chinese_calendar import is_holiday,get_holiday_detail

class DateProcess(object):
    def __init__(self, dateDF):
        self.dateDF = dateDF
    # 判断节假日
    def Get_is_holiday(self, date):
        '''
            获取节假日
        '''
        date = pd.to_datetime(date)
        IsHoliday = is_holiday(date)                      # 是否节假日
        OnHoliday, HolidayName = get_holiday_detail(date) # 是否节假日、节假日名称
        if IsHoliday == True:
            return HolidayName
        else:
            return "Work Day"

    def sales_exact(self,i):
        if i.day == 11 and i.month == 11:
            x = '双11'
        elif i.day == 12 and i.month == 12:
            x = '双12'
        elif i.day == 18 and i.month == 6 :
            x = '618'
        # elif i.day == 18 and i.month == 8:
        #     x = '818'
        else:
            x = ''
        return x

    def sales_range(self,i):
        if datetime.datetime.strptime(str(i.year)+"-"+"11-04", "%Y-%m-%d") < i < datetime.datetime.strptime(str(i.year)+"-"+"11-18", "%Y-%m-%d"):
            x = '双11'
        elif datetime.datetime.strptime(str(i.year)+"-"+"12-05", "%Y-%m-%d") < i < datetime.datetime.strptime(str(i.year)+"-"+"12-19", "%Y-%m-%d"):
            x = '双12'
        elif datetime.datetime.strptime(str(i.year)+"-"+"06-11", "%Y-%m-%d") < i < datetime.datetime.strptime(str(i.year)+"-"+"06-25", "%Y-%m-%d"):
            x = '618'
        # elif datetime.datetime.strptime(str(i.year)+"-"+"08-11", "%Y-%m-%d") < i < datetime.datetime.strptime(str(i.year)+"-"+"08-25", "%Y-%m-%d"):
        #     x = '818'
        else:
            x = ''
        return x
    def scool_leave(self,i):
        if datetime.datetime.strptime(str(i.year) + "-" + "01-15","%Y-%m-%d") < i < datetime.datetime.strptime(str(i.year) + "-" + "02-15", "%Y-%m-%d") :
            x = '寒假'
        elif datetime.datetime.strptime(str(i.year) + "-" + "07-01",  "%Y-%m-%d") < i < datetime.datetime.strptime(str(i.year) + "-" + "09-01", "%Y-%m-%d") :
            x = '暑假'
        else:
            x = ''
        return x

    def add_festival(self,i,day_type):
        if i.day == 14 and i.month == 2:
            day_type = '情人节'
        elif i.day == 8 and i.month == 3:
            day_type = '妇女节'
        elif i.day == 12 and i.month == 5:
            day_type = '母亲节'
        elif i.day == 22 and i.month == 8:
            day_type = '七夕节'

        return day_type

    # 获取日期特征
    def get_Feature_day(self):
        # 年、月、季度、星期
        self.DF = pd.to_datetime(self.dateDF).to_frame(name="time")
        self.DF['year'] = [str(i.year) for i in self.DF['time']]  # 年
        self.DF['month'] = [str(i.month) for i in self.DF['time']]  # 月
        self.DF['quarter'] = [str((int(i) - 1) // 3 + 1) for i in self.DF.month]  # 季度
        self.DF['week'] = [str(i.isoweekday()) for i in self.DF['time']]  # 星期

        # 节假日、调休日
        list_ = [[i, self.Get_is_holiday(i)] for i in self.DF['time']]
        day_type = pd.DataFrame(list_, columns=['day', 'day_type'])
        self.DF['day_type'] = day_type.day_type.fillna("Week Day & Change Holiday").tolist()

        # 促销
        # 准确日期
        self.DF['sales_type_exact'] = self.DF['time'].apply(self.sales_exact)
        # 前后推一周
        self.DF['sales_type_range'] = self.DF['time'].apply(self.sales_range)

        # 学校假期
        self.DF['school_leave'] = self.DF['time'].apply(self.scool_leave)
        dict_day_type_chinese = {
            "New Year's Day":"元旦",
            "Spring Festival": "春节",
            "Tomb-sweeping Day": "清明节",
            "Labour Day": "劳动节",
            "Dragon Boat Festival": "端午节",
            "Mid-autumn Festival": "中秋节",
            "National Day": "国庆节",
            "Week Day & Change Holiday": "周末或调休",
            "Work Day": "工作日"
        }
        self.DF['day_type_chinese'] = [dict_day_type_chinese[i] for i in self.DF.day_type]

        # 补充其他节日
        self.DF['day_type_chinese'] = self.DF.apply(lambda x: self.add_festival(x['time'], x['day_type_chinese']), axis=1)

        return self.DF

    # 获取日期范围
    def get_DateRange(self):
        start_time = datetime.datetime.strptime("2020-01-01", "%Y-%m-%d")
        now = datetime.datetime.now()
        next_month_end = datetime.datetime(now.year, now.month + 2, 1) - timedelta(days=1)  # 下月最后一天
        date_range = pd.date_range(
            start=start_time,  # 开始日期
            end=next_month_end,  # 结束日期
            freq="D",
        )

        return pd.DataFrame(index=date_range)


