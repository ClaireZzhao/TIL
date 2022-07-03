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



