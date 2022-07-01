# TFiDF 가중치 희소행렬 및 문서분류

### 1) csv file 가져오기

```python
import pandas as pd # csv file

path = "some_path"
spam_data = pd.read_csv(path + '/temp_spam_data2.csv',header=None, encoding='utf-8')

print(spam_data.info())
print(spam_data)
'''
0                                                  1
0      ham  Go until jurong point, crazy.. Available only ...
1      ham                      Ok lar... Joking wif u oni...
2     spam  Free entry in 2 a wkly comp to win FA Cup fina...
3      ham  U dun say so early hor... U c already then say...
4      ham  Nah I don't think he goes to usf, he lives aro...
'''
```

### 2) texts, target 전처리

```python
# 1) texts 전처리 : 공백, 특수문자, 숫자
texts = spam_data[1]
print('전처리 전')
print(texts)

# << texts 전처리 함수 >>
import string
def text_prepro(texts):
	# Lower case : 소문자
	texts = [x.lower() for x in texts]
	# Remove punctuation : 문장 부호 제거
	texts = [''.join(ch for ch in st if ch not in string.punctuation) for st in texts]
	# Remove numbers : 숫자 제거
	texts = [''.join(ch for ch in st if ch not in string.digits) for st in texts]
	# Trim extra whitespace : 두 칸 이상 공백 -> 한 칸
	texts = [' '.join(x.split()) for x in texts]
	return texts

# 함수 호출
texts = text_prepro(texts)
print('전처리 후 ')
print(texts)
```

```python
# 2) target 전처리 : dummy변수
target = spam_data[0]
target # spam or ham

target = [1 if t == 'spam' else 0 for t in target]  #list 내포 통한 label encoding
print(target)
```

### 3) max features: 희소행렬의 열 수(word size)

```python
from sklearn.feature_extraction.text import TfidfVectorizer 

obj = TfidfVectorizer() # 단어 생성기
fit = obj.fit(texts)
voca = fit.vocabulary_
print(voca)  # 단어 사전으로 return
'''
{'go': 2873, 'until': 7839, 'jurong': 3780, 'point': 5547, 'crazy': 1567, 'available': 491,...}
'''
len(voca) # 8603
max_features = len(voca) # 8603
```

### 4) sparse matrix : TFiDF 가중치

```python
sparse_mat = obj.fit_transform(texts) # max_features = 8603
print(sparse_mat)
'''
(D, T)       가중치(IFiDF)
(0, 4)	0.4206690600631704
(0, 2)	0.3393931489111758
(0, 9)	0.8413381201263408

(3, 10)	0.40824829046386296
(3, 6)	0.40824829046386296
(4, 15)	0.7782829228046183
(4, 1)	0.6279137616509933
'''

max_features = 5000
'''
max_features = len(voca) : 전체 단어(8603)를 이용해서 희소행렬 
max_features = 5000 : 5,000개 단어만 이용하여 희소행렬 
'''
obj2 = TfidfVectorizer(max_features = max_features,
                       stop_words = 'english')
'''
max_features : 최대 단어 길이 
stop_words : 불용어 단어 제거 
''' 
sparse_mat2 = obj2.fit_transform(texts)
print(sparse_mat2)
sparse_mat2.shape  # (5574, 5000)
```

### 5) numpy array 변환

```python
sparse_mat_arr = sparse_mat2.toarray() # sparse matrix → array
print(sparse_mat_arr) # X 변수로 사용

import numpy as np
y = np.array(target)  # target변수 numpy array 변환 -> y변수
```

### 6) numpy array로 변환된 sparse matrix는 X변수로 사용

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
sparse_mat_arr, y, test_size = 0.3, random_state=123)

X_train.shape # (3901, 5000)
X_test.shape # (1673, 5000)
```

### 7) file save/load

```python
import numpy as np

# file save
spam_train_test = (X_train, X_test, y_train, y_test)
np.save(path + '/spam_train_test.npy', spam_train_test)

# file load
X_train, X_test, y_train, y_test = np.load(path + '/spam_train_test.npy',
allow_pickle = True)

X_train.shape # (3901, 5000)
print(X_train)
```

### 8) NB model로 문서 분류

```python
import numpy as np 
from sklearn.naive_bayes import MultinomialNB # nb model
from sklearn.svm import SVC  # svm model 
from sklearn.metrics import accuracy_score, confusion_matrix # 평가 
import time

nb = MultinomialNB()

chktime = time.time()
model = nb.fit(X = X_train, y = y_train)
chktime = time.time() - chktime
print('실행 시간 : ', chktime)

y_pred = model.predict(X = X_test) # 예측치
y_true = y_test # 관측치

acc = accuracy_score(y_true, y_pred)
print('NB 분류정확도 =', acc)

con_mat = confusion_matrix(y_true, y_pred)
print(con_mat)
'''
실행 시간 :  0.139268159866333
NB 분류정확도 = 0.9719067543335326
[[1435    0]
[  47  191]]
'''
```

### 9) SVM model로 문서분류

```python
svm = SVC(kernel = 'linear')

chktime = time.time()
model2 = svm.fit(X = X_train, y = y_train)
chktime = time.time() - chktime
print('실행 시간 : ', chktime)

y_pred2 = model2.predict(X = X_test)
acc = accuracy_score(y_true, y_pred2)
print('svm 분류정확도 =', acc)

con_mat = confusion_matrix(y_true, y_pred2)
print(con_mat)
'''
실행 시간 :  7.98110294342041
svm 분류정확도 = 0.9790794979079498
[[1433    2]
[  33  205]]
'''
```