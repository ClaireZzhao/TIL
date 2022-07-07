# -*- coding: utf-8 -*-
"""
문1) 최소최대척도(min-max scale)로 변환 후 qsec칼럼의 값이 0.5이상 개수는?
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler # min-max scale 함수

# 1. dataset 가져오기
path = r'some path'
a = pd.read_csv(path +'/mtcars.csv') 
a.info()

# 2. 결과 구하기
qsec = a['qsec']  # 칼럼 추출
qsec_arr = qsec.to_numpy(qsec)  # numpy_array로 변환
qsec_scaled = MinMaxScaler().fit_transform(qsec_arr.reshape(-1,1))  # min-max scaling
result = len(qsec_scaled[qsec_scaled > 0.5])  # 0.5이상 개수


# 문2) 결측치를 포함하는 모든 행을 제거한 후, 상위 70% 데이터에서 mpg컬럼의 제1사분위수를 정수로 출력하시오.   

import pandas as pd
path = r'some path'
a = pd.read_csv(path +'/mtcars.csv')  

df = a.copy()
   
# 1) 결측치를 포함하는 모든 행을 제거
df.isnull().sum()
'''
mpg     1
disp    2
wt      1
'''

# 전체 칼럼 결측치 제거 
df = df.dropna()
df.shape # (29, 11)

# 2) 상위 70% 데이터
len(df) # 29
size = int(len(df) * 0.7) # 20 
idx = range(size) # 0 ~ 19

top70 = df.iloc[idx, ]
top70.shape # (20, 11)

# 3) mpg컬럼의 제1사분위수
des = top70['mpg'].describe()

des.index
'''
Index(['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'], dtype='object')
'''
des.values 
'''
array([20.        , 20.685     ,  6.04285354, 10.4       , 17.075     ,
       20.1       , 22.8       , 33.9       ])
'''

result = int(des['25%']) # 정수 변환 
print(result) # 17

