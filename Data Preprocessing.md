# 데이터 전처리(Data Preprocessing)

## 머신러닝 모델의 예측력을 향상시키기 위해 데이터 전처리가 필요함

### 1. 결측치(NaN) 처리:  결측치 확인 & 처리(제거 또는 대체)

- 결측치 확인

```python
df.isnull().any()  # 전체 칼럼 기준으로 결측치 유무 확인: default: axis = 0(열)
df.isnull().any(axis=1) # 전체 행 기준으로 결측치 유무 확인: return true/false

df.isnull().sum()  # 결측치 개수 확인 
```

- 결측치 제거: 결측치가 적은 경우

```python
dropna = df.dropna(subset=['칼럼명']) # 결측치 포함 행 제거(특정칼럼 기준)
dropna2 = df.dropna()  # 결측치 포함 행 제거(전체 칼럼 기준)
```

- 결측치 대체: 결측치가 많은 경우

```python
df['칼럼명'] = df['칼럼명'].fillna(0)  # 0으로 대체
df['칼럼명'].fillna(df['칼럼명'].mean(), inplace=True) # 평균으로 대체

data_null_value = data.fillna(value={'Horsepower': data['Horsepower'].mode()[0],
                           'Miles_per_Gallon': data['Miles_per_Gallon'].mean()}, 
                            inplace = False) # 여러 칼럼에 있는 결측치 대체
```

<aside>
👉 중복된 값이 있는 경우

</aside>

```python
data.duplicated()  # 중복된 값이 있으면 True로, 아니면 False로 return 
data[data.duplicated()] # 색인으로 탐색
data.duplicated().sum() # 중복된 값은 몇개가 있는지 return

data.drop_duplicates() # 첫번째 값을 제외한 중복 값을 제거(전체 칼럼 기준)
data.drop_duplicates(subset=['Horsepower', 'Miles_per_Gallon'], keep='first', 
                     inplace = False) # 특정 칼럼 가준
```

### 2. 이상치(outlier) 처리: 이상치 탐색 & 이상치 처리(제거 또는 대체)

- 이상치: 정상범주에서 벗어난 값(극단적으로 크거나 작은 값)
- 정상범주를 어떻게 확인하는가?
    
    <aside>
    👉 하한값 = 평균 - n*표준편차 (n = 3: threshold)
    상한값 = 평균 + n*표준편차 (n = 3: threshold)    0
    
    </aside>
    
    <aside>
    👉 minval = Q1 -  IQR * 1.5         # IQR = Q3 - Q1 
    maxval = Q3 + IQR * 1.5
    
    </aside>
    
- 범주형 변수의 경우:

```python
# 이상치 확인
dataset.gender.unique() # unique()함수로 확인
dataset.gender.value_counts() # value_counts()함수로 확인
plt.pie(dataset.gender.value_counts()) # pie chart로 확인
plt.show()
```

```python
# 이상치 처리:
data = data[data.gender == 1 | data.gender == 2] # subset통해서 이상치 제거

# 이상치 제거: 관측치가 적고, 해당 변수의 의미를 일고 있는 경우
```

- 연속형 변수의 경우:

```python
# 요약통계량 보기
data.describe() # 최댓값 문제 or 음수값 문제 등 확인 가능

# boxplot 이용 이상치 탐색
import matplotlib.pyplot as plt

plt.boxplot(data['칼럼명'])
plt.show()

# 정상범위를 구한 경우(IQR 수식 등 이용하여)
data[(data['age'] < minval) | (data['age'] > maxval)] # 이상치 확인 
```

```python
# 이상치 제거
new_data = data[data.bmi > 0] # subset 이용

# 이상치 대체: 정상범주의 하한값과 상한값으로 대체
X.loc[X.Fare > maxval, 'Fare'] = maxval # 상한값으로 대체
```

<aside>
👉 model 만드는데 이상치를 제거하거나 대체하면 되는데, 
업무에 있어서 이상치가 많은 가치들을 가질 수 있으므로 이상치 처리하기 전에 확인 필요함

</aside>

### 3. 데이터 인코딩(data encoding)

- 데이터 인코딩 대상: machine learning model에서 **범주형 변수** 대상으로 숫자형의 목록으로 변환해주는 전처리 작업
- 데이터 인코딩 방법: label encoding, one-hot encoding
    - label encoding: y변수 or 트리모델 계열의 x변수 대상(ex: no/yes → 0/1) -  10진수로 인코딩
        
        ```python
        from sklearn.preprocessing import LabelEncoder
        
        encoder = LabelEncoder() # 객체 생성(Init)
        labels = encoder.fit_transform(df['칼럼명']) # 영문자 오름차순
        ```
        
    - one-hot encoding: 회귀모델, SVM 계열의 x변수 대상 - 2진수(더미변수)로 인코딩
        
        ** 회귀모델에서 인코딩값이 가중치로 적용되므로 x변수를 one-hot encoding로 변환
        
        ```python
        import pandas as pd
        df_dummy = pd.get_dummies(data=df) #k개 가변수(더미변수)생성-기준변수(base)포함
        df_dummy2 = pd.get_dummies(data=df, drop_first=True) # base 제외**(권장)**
        
        #숫자형변수는 자동으로 제외됨(object형변수 대상만)->숫자형변수 대상으로 encoding 필요시
        df_dummy3 = pd.get_dummies(data=df, drop_first=True, 
                                   columns = ['칼럼명1', '칼럼명2', '칼럼명3']) 
        
        # 더미변수의 base를 변경하고자 할 때
        # 1) object형 -> category형 변환
        iris['species'] = iris['species'].astype('category')
        iris['species'] = iris['species'].cat.set_categories(['versicolor',
                                                              'virginica',
                                                              'sentosa'])
        iris_dummy = pd.get_dummies(data=iris, columns=['species'],
                                    drop_first=True)
        ```
        

### 4. 피처 스케일링(feature scaling)

- X변수(feature) 대상: 서로 다른 크기(단위)를 일정한 범위로 조정하는 전처리 작업
- 방법: 표준화, 최소-최대 정규화, 로그변환
    - 표준화(StandardScaler): x변수 대상으로 정규분포가 될 수 있도록 평균=0, 표준편차=1로 통일
        
        <aside>
        👉 회귀모델, SVM계열에 적용: x변수가 정규분포라고 가정하에 학습 진행
        
        </aside>
        
        ```python
        from sklearn.preprocessing import scale # 표준화(mu=1, std=1)
        x_zscore = scale(x)
        
        from sklearn.preprocessing import StandardScaler
        x_scaled = StandardScaler().fit_transform(X=iris.drop('species', axis=1))
        ```
        
    - 최소-최대 정규화(MinMaxScaler): x변수 대상으로 최솟값=0, 최댓값=1로  통일
        
        <aside>
        👉 트리모델 계열(회귀모델 계열이 아닌 경우)에 적용
        
        </aside>
        
        ```python
        from sklearn.preprocessing import minmax_scale # 정규화(0~1)
        x_nor = minmax_scale(x)
        
        from sklearn.preprocessing import MinMaxScaler
        x_scaled = MinMaxScaler().fit_transform(X=iris.drop('species', axis=1))
        ```
        
    - 로그변환(log-transformation): log()함수로 비선형을 선형으로, 왜곡을 갖는 분포를 죄우대칭 형태의 정규분포로 변환
        
        <aside>
        👉 회귀모델에서 y변수 적용
        
        </aside>
        
        ```python
        import numpy as np
        x_log = np.log(x)
        x_log1 = np.log1p(x) # x변수에 0이 있는 경우(warning메시지)
        x_log2 = np.log(np.abs(x)) # x변수에 음수가 있는 경우(warning메시지)
        ```
        
- 어떤 방법으로 스케일링 하면 좋은지?
    
    ```python
    from sklearn.preprocessing import minmax_scale, scale
    import numpy as np
    
    def scaling(X, y, kind='none') : 
        # x변수 스케일링  
        if kind == 'minmax_scale' :  
            X_trans = minmax_scale(X) 
        elif kind == 'zscore' : 
            X_trans = scale(X)  # zscore(X)
        elif kind == 'log' :  
            X_trans = np.log1p(np.abs(X)) 
        else :
            X_trans = X 
        
        # y변수 로그변환 
        if kind != 'none' :
            y = np.log1p(np.abs(y))   
        
        # train/test split 
        X_train,X_test,y_train,y_test = train_test_split(
            X_trans, y, test_size = 30, random_state=1)   
        
        print(f"scaling 방법 : {kind}, X 평균 = {X_trans.mean()}")
        return X_train,X_test,y_train, y_test
    
    # 함수 호출 
    #X_train,X_test,y_train,y_test = scaling(X, y,'none')
    #X_train,X_test,y_train,y_test = scaling(X, y,'minmax_scale')
    #X_train,X_test,y_train,y_test = scaling(X, y,'zscore')
    X_train,X_test,y_train,y_test = scaling(X, y,'log')
    
    # model 생성하기
    model = LinearRegression().fit(X=X_train, y=y_train)  
    
    # model 평가하기
    model_train_score = model.score(X_train, y_train) 
    model_test_score = model.score(X_test, y_test) 
    print('model train score =', model_train_score)
    print('model test score =', model_test_score)
    
    y_pred = model.predict(X_test)
    y_true = y_test
    print('R2 score =',r2_score(y_true, y_pred))  
    mse = mean_squared_error(y_true, y_pred)
    print('MSE =', mse)
    
    '''
    1. 기본: x, y변수 스케일링 전
    scaling 방법 : none, X 평균 = 70.07396704469443
    model train score = 0.7410721208614651
    model test score = 0.7170463430870563
    R2 score = 0.7170463430870563
    MSE = 20.20083182974776
    
    2. X: 정규화, y: 로그변환
    scaling 방법 : minmax_scale, X 평균 = 0.3862566314283195
    model train score = 0.7910245321591743
    model test score = 0.7633961405434472
    R2 score = 0.7633961405434472
    MSE = 0.027922682660046286
    [해설] 정확도 향상, MSE: 0 수령정도 평가
    
    3. X: 표준화, y: 로그변환
    scaling 방법 : zscore, X 평균 = -1.1147462804871136e-15
    model train score = 0.7910245321591743
    model test score = 0.7633961405434472
    R2 score = 0.7633961405434472
    MSE = 0.027922682660046282
    [해설] 정규화와 정확도 결과는 동일함
    
    4. X: 로그변환, y: 로그변환
    scaling 방법 : log, X 평균 = 2.408779096889065
    model train score = 0.7971243680501752 (test score보다 더 높아야 하는데..)
    model test score = 0.8217362767578845
    R2 score = 0.8217362767578845
    MSE = 0.021037701520680126
    [해설] test score가 높은 불완전한 결과
    '''
    ```