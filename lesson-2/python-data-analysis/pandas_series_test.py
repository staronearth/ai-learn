import pandas as pd
import numpy as np

def create_series():
    #列表创建
    list_series=pd.Series([1,2,3,4,5])
    print(list_series)
    #字典创建
    dict_series=pd.Series({'a':1,'b':2,'c':3})
    print(dict_series)
    #制定索引和数据类型
    new_series=pd.Series(data=[4,5,6],index=['e','f','g'],dtype=pd.Float64Dtype)
    print(new_series)

def property_series():
    s=pd.Series(data=[4,5,6],index=['e','f','g'],dtype=pd.Float64Dtype,name="成绩")
    print(s.dtype)
    print(s.index)
    print(s.values)
    print(s.shape)
    print(s.hasnans)
    print(s.name)

def series_data_find():
    s=pd.Series(data=[4,5,7,8,np.nan,10],
    index=['a','b','c','d','e','f'],dtype=pd.Float64Dtype,name="序列")
    print(s)
    print(s.iloc[0])
    print(s.loc['a'])
    print(s.iat[1])
    print(s.at['b'])

    print(s[0:3])
    print(s['a':'c'])
    
def data_clear():
    s=pd.Series(data=[4,5,7,8,np.nan,10,4,5,8],
    index=['a','b','c','d','e','f','g','h','j'],dtype=pd.Float64Dtype,name="序列")

    print(s.describe())
    print(s.dropna())

    print(s.fillna(0))

    print(s.ffill())
    print(s.bfill())

    print(s.drop_duplicates())

    print(s[s>=10])
    print(s.isin([4,8]))
    print(s.get('z',0))

def data_change():
    s=pd.Series(data=[4,5,7,8,np.nan,10,4,5,8],
    index=['a','b','c','d','e','f','g','h','j'],dtype=pd.Float64Dtype,name="序列")
    print(s.apply(lambda x:x**2))
    print(s.map({4:44,5:55}))
    print(s.replace({4:44,5:55}))

def series_sort():
    s=pd.Series(data=[4,5,7,8,np.nan,10,4,5,8],
    index=['a','b','c','d','e','f','g','h','j'],dtype=pd.Float64Dtype,name="序列")

    print(s.sort_index())
    print(s.sort_values())

def series_count_windos_cnt():
    s = pd.Series(data=[100, 95, 102, 88, 110, 105, 98, 112, 115], 
              index=['a1', 'b1', 'c1', 'a2', 'b2', 'c2', 'a3', 'b3', 'c3'],dtype=pd.Float64Dtype,name="营业值")

    print(s.describe())
    print(s.rolling(window=3).sum())
    print(s.expanding(min_periods=3).sum())
    print(s.cumsum())

if __name__=="__main__":
    # create_series()
    # property_series()
    # series_data_find()
    # data_clear()
    # data_change()
    # series_sort()
    series_count_windos_cnt()