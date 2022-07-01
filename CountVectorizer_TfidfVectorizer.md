# CountVectorizer & TfidfVectorizer

### CountVectorizer 가중치 vs TfidfVectorizer 가중치

sparse matrix(자연어를 숫자로 변환)를 구성하는 과정에서 주로 단어빈도수를 기반으로 한 CountVectorizer및 단어 빈도수-역 문서 빈도수를 기반으로 한 TfidfVectorizer 많이 쓴다.

CountVectorizer를 이용하여 sparse matrix를 구성하는 코드는 아래와 같다.

```python
sample = ["Machine learning is fascinating, it is wonderful",
          "Machine learning is a sensational technology",
          "Elsa is a popular character"]

from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer()

X = vec.fit_transform(sample)
X # (3 * 11)

vec.get_feature_names_out()  # 11개
'''
['character',
 'elsa',
 'fascinating',
 'is',
 'it',
 'learning',
 'machine',
 'popular',
 'sensational',
 'technology',
 'wonderful']
'''

import pandas as pd
pd.set_option('display.max_columns', 50)

CVresult = pd.DataFrame(X.toarray(), columns = vec.get_feature_names())
CVresult
'''
      character  elsa  fascinating  is  it  learning  machine  popular  \
0          0     0            1      2   1         1        1        0   
1          0     0            0      1   0         1        1        0   
2          1     1            0      1   0         0        0        1   

   sensational  technology  wonderful  
0            0           0          1  
1            1           1          0  
2            0           0          0  
'''
# is의 빈도수가 가장 높게 나왔습니다.
```

실행결과에 따르면 ‘is’라는 단어가 4번이나 가장 많이 출현했다. 이 의미는 해당 단어가 문서상에서 출현되는 확률이 가장 크다는 것이다. 하지만 ‘is’라는 단어가 문장 해석에 유의미한 정보는 아니다. 

따라서, 단어출현빈도수를 가중치로 적용하면, ‘is’와 비슷한 단어들의 출현빈도수가 높게 나올 수 있고, 잘못된 결과를 도출할 수 있다. 이를 해결하기 위해서, 단어출현빈도수를 가중치로 적용하는 것보다, 문장에 차지하는 비율을 적용하여 단어를 인코딩하는 방법 즉 TFiDF(term frequency-inverse document frequency) 가중치를 적용하여 sparse matrix를 구성하면 자주 나타나는 의미없는 단어들을 억제할 수 있다.

TfidfVectorizer를 이용하여 sparse matrix를 구성하는 코드는 아래와 같다.

```python
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF

vec = TFIDF()
vec = vec.fit(sample)
vec.get_feature_names_out()
'''
['character',
 'elsa',
 'fascinating',
 'is',
 'it',
 'learning',
 'machine',
 'popular',
 'sensational',
 'technology',
 'wonderful']
'''
X = vec.fit_transform(sample)
X # 3x11

TFIDFresult = pd.DataFrame(X.toarray(), columns = vec.get_feature_names())
TFIDFresult
'''
   character      elsa  fascinating        is        it  learning   machine  \
0   0.000000  0.000000     0.424396  0.501310  0.424396  0.322764  0.322764   
1   0.000000  0.000000     0.000000  0.315444  0.000000  0.406192  0.406192   
2   0.546454  0.546454     0.000000  0.322745  0.000000  0.000000  0.000000   

    popular  sensational  technology  wonderful  
0  0.000000     0.000000    0.000000   0.424396  
1  0.000000     0.534093    0.534093   0.000000  
2  0.546454     0.000000    0.000000   0.000000  
'''

# TFIDF로 인코딩 후, 출현빈도수가 높은 단어들의 가중치가 작아진건가요?

CVresult.sum(axis=0)/CVresult.sum(axis=0).sum() # 단어출현비율
'''
character      0.0625
elsa           0.0625
fascinating    0.0625
is             0.2500
it             0.0625
learning       0.1250
machine        0.1250
popular        0.0625
sensational    0.0625
technology     0.0625
wonderful      0.0625
'''

TFIDFresult.sum(axis=0)/TFIDFresult.sum(axis=0).sum() # 단어출현비율
'''
character      0.083071
elsa           0.083071
fascinating    0.064516
is             0.173225
it             0.064516
learning       0.110815
machine        0.110815
popular        0.083071
sensational    0.081192
technology     0.081192
wonderful      0.064516
'''
```

의미없는 단어들을 기준으로 보았을 때, CountVectorizer 가중치를 적용한 결과값에 대비해서, TfidfVectorizer 가중치를 적용한 결과값에서 단어출현비율이 줄어든 것을 확인할 수 있다.

반대로 의미있는 단어들의 단어출현비율이 향샹된 것을 확인할 수 있다.