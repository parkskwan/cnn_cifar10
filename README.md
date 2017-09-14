# Convolution neural network cifar10
Canadian Institute for Advanced Research (CIFAR, pronounced "see far")

CIFAR-10은 컬러이미지 6만개가 10종류로 구성되어 있다. 5만개는 인식훈련용이고 1만개는 시험용이다. 비행기, 자동차, 새, 고양이, 사슴, 개, 개구리, 말, 배, 트럭 등을 무작위 이미지로 학습하도록 되어 있다. CIFAR-100은 600개 이미지씩 100종의 이미지로 구성되어 있다. 해상 동물, 물고기, 꽃, 식품 용기, 과일과 채소, 가전기기, 가구, 벌레, 야생동물, 자연풍경, 사람, 파충류, 나무, 자동차 등 다양한 소재를 다룬다. 훈련용이 종별로 500개씩, 시험용은 100개씩 편성하고 있다. 최신 기술의 오류 발생률이 11.2% 정도다. 이렇게 훈련된 소프트웨어가 가려낼 수 있는 물체의 종류는 100가지로 한정된다. 

## 개발 목표
컴퓨터가 사진을 문장으로 설명한다


사람들은 복잡한 장면을 보아도 간단히 중요한 핵심을 끄집어내 설명하지만(?), 컴퓨터로선 무엇이 핵심인지 구분해내기가 쉽지 않다. 그런데 최근 주목할 만한 인공지능 기술이 개발됐다. 구글이 개발한 기계학습기능은 사진에 나온 장면을 문장으로 설명하는 지능 기술이다. 물체의 경계들을 감지하여 분류하고 이름을 붙여 배치를 설명해내는 수준에 이르렀다. 이 설명은 사진이 갖는 핵심 의미를 찾아서 서술해준다. 구글이 이미지를 해석하는 데 적용한 기술은 심화공진화신경망(深化共進化神經網, Deep Covolutional Neural Network, CNN)이다. 이미 알고 있는 물체들이 이미지 속에 있을 확률을 계산해서 물체를 확인해 주는 기술이다. 이때 확인된 물체들의 상관관계를 문장으로 설명하는 기술은 자동번역기술에서 활용해온 재귀열 신경망(RNN) 기술이다. 예를 들면, 한국어 문장을 언어 벡터공간에서 같은 빈도로 사용되는 단어들을 모아 연결한 벡터로 표현한 다음에 통째로 프랑스 문장으로 바꾸는 방식이다. 서로 다른 언어라 할지라도 주어진 환경에서 나누는 말이 같기 때문이다. 예를 들어 식탁에 둘러앉아 있는데 상석에 앉은 사람이 “자, 듭시다”고 말하며 식사를 시작했다고 하면, 프랑스 문화권에선 같은 상황에서 어떤 말이 사용되는지를 찾아서 “Bon appetite”라고 통째로 바꿔주는 방법이다. 이미지 속의 물체들을 확인해내는 CNN법과 이들의 관계를 서술해 주는 RNN법을 결합해서 어떤 사진을 보더라도 바로 문장으로 설명해줄 수 있는 기술을 완성했다.



```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.datasets import cifar10

from keras.utils import np_utils

import h5py

import numpy as np
import pandas as pd
```

```
cifar10_w = 32; cifar10_h = 32; cifar10_c = 3; 

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.reshape(X_train.shape[0], cifar10_w, cifar10_h, cifar10_c).astype('float32') / 255
X_test  = X_test.reshape(X_test.shape[0], cifar10_w, cifar10_h, cifar10_c).astype('float32') / 255

print(X_train.shape); print(X_test.shape)
```

```python
%matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.misc import toimage

nclasses = 10; num = 5; pos = 1

categories = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

plt.figure(figsize=(17, 17), frameon=False)


for targetClass in range(nclasses):
    targetIdx = []
   
    for i in range(len(y_train)):
        if y_train[i][0] == targetClass:
            targetIdx.append(i)

    np.random.shuffle(targetIdx)
    for idx in targetIdx[:num]:
        img = toimage(X_train[idx])
        plt.subplot(nclasses, num, pos)
        plt.imshow(img, interpolation='nearest')
        plt.title("%s" % (categories[targetClass]), fontsize=11, loc='left')
        plt.subplots_adjust(wspace=0.1, hspace=0.4)  
        plt.axis('off')
        pos += 1

plt.show()
```

```python
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping

nfilter = 32; bsize = 200; nb_classes = 10; opt = ['adam','rmsprop']

model = Sequential()

model.add(Conv2D(nfilter, (3, 3), padding="same", input_shape = X_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(2*nfilter, (3, 3), padding="same"))
model.add(Activation('relu'))
model.add(Conv2D(2*nfilter, (3, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy', optimizer=opt[1], metrics=['accuracy'])

```

