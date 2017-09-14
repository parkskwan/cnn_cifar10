# Convolution neural network cifar10
Canadian Institute for Advanced Research (CIFAR, pronounced "see far")

CIFAR-10은 컬러이미지 6만개가 10종류로 구성되어 있다. 5만개는 인식훈련용이고 1만개는 시험용이다. 비행기, 자동차, 새, 고양이, 사슴, 개, 개구리, 말, 배, 트럭 등을 무작위 이미지로 학습하도록 되어 있다. CIFAR-100은 600개 이미지씩 100종의 이미지로 구성되어 있다. 해상 동물, 물고기, 꽃, 식품 용기, 과일과 채소, 가전기기, 가구, 벌레, 야생동물, 자연풍경, 사람, 파충류, 나무, 자동차 등 다양한 소재를 다룬다. 훈련용이 종별로 500개씩, 시험용은 100개씩 편성하고 있다. 최신 기술의 오류 발생률이 11.2% 정도다. 이렇게 훈련된 소프트웨어가 가려낼 수 있는 물체의 종류는 100가지로 한정된다. 

## 개발 목표
컴퓨터가 사진을 문장으로 설명한다


사람들은 복잡한 장면을 보아도 간단히 중요한 핵심을 끄집어내 설명하지만(?), 컴퓨터로선 무엇이 핵심인지 구분해내기가 쉽지 않다. 그런데 최근 주목할 만한 인공지능 기술이 개발됐다. 구글이 개발한 기계학습기능은 사진에 나온 장면을 문장으로 설명하는 지능 기술이다. 물체의 경계들을 감지하여 분류하고 이름을 붙여 배치를 설명해내는 수준에 이르렀다. 이 설명은 사진이 갖는 핵심 의미를 찾아서 서술해준다. 구글이 이미지를 해석하는 데 적용한 기술은 심화공진화신경망(深化共進化神經網, Deep Covolutional Neural Network, CNN)이다. 이미 알고 있는 물체들이 이미지 속에 있을 확률을 계산해서 물체를 확인해 주는 기술이다. 이때 확인된 물체들의 상관관계를 문장으로 설명하는 기술은 자동번역기술에서 활용해온 재귀열 신경망(RNN) 기술이다. 예를 들면, 한국어 문장을 언어 벡터공간에서 같은 빈도로 사용되는 단어들을 모아 연결한 벡터로 표현한 다음에 통째로 프랑스 문장으로 바꾸는 방식이다. 서로 다른 언어라 할지라도 주어진 환경에서 나누는 말이 같기 때문이다. 예를 들어 식탁에 둘러앉아 있는데 상석에 앉은 사람이 “자, 듭시다”고 말하며 식사를 시작했다고 하면, 프랑스 문화권에선 같은 상황에서 어떤 말이 사용되는지를 찾아서 “Bon appetite”라고 통째로 바꿔주는 방법이다. 이미지 속의 물체들을 확인해내는 CNN법과 이들의 관계를 서술해 주는 RNN법을 결합해서 어떤 사진을 보더라도 바로 문장으로 설명해줄 수 있는 기술을 완성했다.

## 미완성된 부분...
CNN 분류기에 LSTM를 적용해야 완성된다. 따라서 지금은 cifar10 분류작업에 만족해야 합니다. 그러나 빠른 시일 내에 LSTM를 적용하여 컴퓨터가 사진을 문장으로 설명하는 프로그램을 만드는 것을 목표로 하겠습니다.   


지금 부터 cifar10 분류하는 프로그램을 만들어 보겠습니다.

### cifar10 불러 드리기 위한 import 루틴
```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.datasets import cifar10

from keras.utils import np_utils

import h5py

import numpy as np
import pandas as pd
```

### cifar10를 CNN 에서 트레이닝 시키기 위한 전처리 작업

2차원 (32 X 32), RGB(3) 그리고 가장 큰 값이 1.0이 되도록 하는 루틴    

```
cifar10_w = 32; cifar10_h = 32; cifar10_c = 3; 

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.reshape(X_train.shape[0], cifar10_w, cifar10_h, cifar10_c).astype('float32') / 255
X_test  = X_test.reshape(X_test.shape[0], cifar10_w, cifar10_h, cifar10_c).astype('float32') / 255

print(X_train.shape); print(X_test.shape)
```

### cifar10을 제대로 load했는지 그림을 plot 해서 알아보는 루틴

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

### CNN의 MLP를 만들어 주는 루틴

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

print(model.summary())
```
### CNN의 MLP가 제대로 만들어 졌는지 확인

```
# print(model.summary()) : 출력되는 결과
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_16 (Conv2D)           (None, 32, 32, 32)        896       
_________________________________________________________________
activation_19 (Activation)   (None, 32, 32, 32)        0         
_________________________________________________________________
max_pooling2d_11 (MaxPooling (None, 16, 16, 32)        0         
_________________________________________________________________
dropout_16 (Dropout)         (None, 16, 16, 32)        0         
_________________________________________________________________
conv2d_17 (Conv2D)           (None, 16, 16, 64)        18496     
_________________________________________________________________
activation_20 (Activation)   (None, 16, 16, 64)        0         
_________________________________________________________________
conv2d_18 (Conv2D)           (None, 14, 14, 64)        36928     
_________________________________________________________________
max_pooling2d_12 (MaxPooling (None, 7, 7, 64)          0         
_________________________________________________________________
dropout_17 (Dropout)         (None, 7, 7, 64)          0         
_________________________________________________________________
flatten_6 (Flatten)          (None, 3136)              0         
_________________________________________________________________
dense_9 (Dense)              (None, 512)               1606144   
_________________________________________________________________
activation_21 (Activation)   (None, 512)               0         
_________________________________________________________________
dropout_18 (Dropout)         (None, 512)               0         
_________________________________________________________________
dense_10 (Dense)             (None, 10)                5130      
_________________________________________________________________
activation_22 (Activation)   (None, 10)                0         
=================================================================
Total params: 1,667,594
Trainable params: 1,667,594
Non-trainable params: 0
_________________________________________________________________
None
```

### cifar10-model의 train 결과가 있으면 그것을 load하고 없으면 train set을 적용하여 계산하는 작업

```
hdf5_file="./cifar10-model.hdf5"

if os.path.exists(hdf5_file):
    model.load_weights(hdf5_file)
else:
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=bsize, epochs=5)
    model.save_weights(hdf5_file)
```

### cifar10-model의 결과을 적용하여 test set을 계산하는 작업

```
score = model.evaluate(X_test, Y_test, batch_size=bsize)
print("\n\n\n\nloss =", score[0], ", accuracy =", score[1],", baseline error: %.2f%%" % (100-score[1]*100))
```
생각보다 상당히 괜찮은 결과가 나왔다.

계산된 결과 : loss = 0.142800596654 , accuracy = 0.944240015745 , baseline error: 5.58%



### 잘못 예측한 결과를 찾아서 plot하는 작업을 해야 하는 이유

잘못 예측한 결과를 찾아 plot 한 다음 무엇이 문제인지 고민 해야 하는 가장 중요한 작업이 남아있습니다. 그러나 여기서는 더 이상 진행은 하지 않고 plot만 하겠습니다.

```
y_pred = model.predict_classes(X_test)

true_preds = [(x,y) for (x,y,p) in zip(X_test, y_test, y_pred) if y == p]
false_preds = [(x,y,p) for (x,y,p) in zip(X_test, y_test, y_pred) if y != p]
print("Number of true predictions: ", len(true_preds))
print("Number of false predictions:", len(false_preds))


nb_classes = 10

plt.figure(figsize=(8, 8))
for i,(x,y,p) in enumerate(false_preds[0:15]):
    plt.subplot(3, 5, i+1)
    plt.imshow(x, interpolation='nearest')
    plt.title("y: %s\np: %s" % (categories[int(y)], categories[int(p)]), fontsize=11, loc='left')
    plt.axis('off')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)  
    
```

### train된 filters을 plot하는 작업

보통은 filter를 plot해서 무언가을 알 수 있지는 않다. 그러나 사람들은 train된 filters를 보고 싶어 한다.
```
from keras import backend as K

W = model.layers[0].get_weights()[0]

print(W.shape)
if K.image_dim_ordering() == 'tf':
    # (nb_filter, nb_channel, nb_row, nb_col)
    W = W.transpose(3, 2, 0, 1)

    nb_filter, nb_channel, nb_row, nb_col = W.shape
    print(W.shape)

    
#plt.figure()
plt.figure(figsize=(10, 10), frameon=False)
for i in range(32):
        im = W[i]
        plt.subplot(6, 6, i + 1)
        plt.axis('off')
        plt.imshow(im)
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()
```
작성중....
