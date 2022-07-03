'''
문2) 아래 백화점 고객의 1년간 구매 데이터이다. 성별 예측모형을 만든 후 이를 평가용
데이터셋에 적용하여 얻은 고객의 성별 예측값(남자일 확률)을 다음과 같은 형식의
csv 파일로 생성하시오.
custid, gender
3500, 0.267
3501, 0.578
3502, 0.885
'''
pd.set_option('display.max_columns', 20)

# 1. dataset 가져오기
path = r'some path'
X_train = pd.read_csv(path + '/X_train.csv', encoding='euc-kr')
X_test = pd.read_csv(path + '/X_test.csv', encoding='euc-kr')
y_train =pd.read_csv(path + '/y_train.csv', encoding='euc-kr')

# 2. 변수 탐색
#
# 1) 자료형 체크
X_train.info()
X_train['주구매상품'].unique()   # 인코딩 필요
X_train['주구매지점'].unique()   # 인코딩 필요

y_train.info()
y_train['gender'].unique()

X_test.info()
X_test['주구매상품'].unique()  # 인코딩 필요
X_test['주구매지점'].unique()  # 인코딩 필요

# 2) 결측치 확인
X_train.isnull().sum()   # 환불금액  2295 결측치 처리 필요
X_test.isnull().sum()    # 환불금액  1611 결측치 처리 필요

# 3) 이상치 확인
X_train.describe()  # 총구매액 & 최대구매액의 최소값이 음수값인걸 발견
X_test.describe()   # 총구매액 & 최대구매액의 최소값이 음수값인걸 발견

# 4) 스케일링 필요한지 확인: 각 변수의 척도가 다르므로 스케일링 필요

# 3. 데이터 전처리(인코딩, 결측치 처리, 이상치 처리, 스케일링)

# 먼저, X_train & y_train 합쳐서 train set로 만든 후, 필요없는 cust_id 삭제하기
train_full = pd.merge(left= X_train, right = y_train, on='cust_id')
train_full.info()  # 3500, 11
train_full = train_full.drop(['cust_id'], axis=1)

cust_id = X_test['cust_id']   # cust_id 보관
X_test = X_test.drop(['cust_id'], axis=1)

# 인코딩(훈련셋, 테스트셋)
from sklearn.preprocessing import LabelEncoder
train_full['주구매상품'] = LabelEncoder().fit_transform(train_full['주구매상품'])
train_full['주구매지점'] = LabelEncoder().fit_transform(train_full['주구매지점'])

X_test['주구매상품'] = LabelEncoder().fit_transform(X_test['주구매상품'])
X_test['주구매지점'] = LabelEncoder().fit_transform(X_test['주구매지점'])

# 결측치 처리(훈련셋 & 테스트셋)
train_full['환불금액'] = train_full['환불금액'].fillna(0)
X_test['환불금액'] = X_test['환불금액'].fillna(0)

# 이상치 처리(훈련셋의 이상치는 삭제해도 되지만, 테스트셋의 이상치는 대체하는 게 좋음)
train_full = train_full[train_full['총구매액'] >= 0]   # 이상치 삭제

X_test.loc[X_test['총구매액'] < 0, '총구매액'] = 0  # 이상치 대체
X_test.loc[X_test['최대구매액'] < 0, '최대구매액'] = 0  # 이상치 대체

# 스케일링(훈련셋 & 테스트셋)
from sklearn.preprocessing import MinMaxScaler
X_train = train_full.drop(['gender'], axis=1)
y_train = train_full['gender']

X_train = MinMaxScaler().fit_transform(X_train)
X_test = MinMaxScaler().fit_transform(X_test)

# 예측 모델 생성
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100,
                               random_state=123).fit(X=X_train, y=y_train)

train_score = model.score(X_train, y_train)
print(train_score)

# model 테스트 : 예측치 구하기
y_pred = model.predict_proba(X = X_test) # 확률 예측

# 8. 결과 도출
result = pd.DataFrame({'custid':cust_id, 'gender':y_pred[:,1]},
             columns=['custid', 'gender']) # custid, gender

print(result)

result.to_csv(path + '/0001234.csv', index=False) # 행이름 제외

"""
1. 범주형(Categorical) 변수 Encoding
 - 기계학습을 위해서 범주형 변수를 숫자로 바꿔주는 작업 

2. 인코딩(encoding) 유형과 적용 사례 
1) One-Hot Encoding : 각 범주(목록)를 대상으로 2진수로 변환하는 방법
 - 일반적으로 x변수를 대상으로 인코딩 
 - 회귀계열 모델(로지스틱회귀, SVM, 신경망)에서 x변수 인코딩에 이용 
   예) 혈액형 : A, AB, B, O,  -> 0 0 0, 1 0 0, 0 1 0, 0 0 1   

2) Label Encoding : 각 범주(목록)를 대상으로 10진수로 변환하는 방법
 - 일반적으로 y변수(대상변수)를 대상으로 인코딩 
 - 트리계열 모델(의사결정트리, 랜덤포레스트)에서 x변수 인코딩에 이용
"""