import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

import keras
from tensorflow import keras
import matplotlib.pyplot as plt

from keras.layers import Dense, Embedding, GRU, Bidirectional, Input
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

class PredictDot:
    '''Прогноз по трем предыдущим отчетам и трем последующим отчетам
    с использованем двунаправленной рекурентной сети.
    Данный прогноз будет строиться на основе предыдущего контекста и
    на основе будущего контекста.'''

    # задаем количество наблюдений(точек) для синусоиды
    COUNT_DOTS = 10000

    # определяем сколько отчетов до и после мы будем брать:
    DIAPAZON = 3
    # всего у нас будет 7 отчетов:
    TOTAL_LENGTH = DIAPAZON * 2 + 1

    def build_sinusoid(self):
        '''Формируем синусоиду со случайным шумом'''


        #определяем синусоиду и добавляем шум:
        data=np.array([np.sin(x/20) for x in range(self.COUNT_DOTS)])+\
             0.1*np.random.randn(self.COUNT_DOTS)

        #построим график синусоиды:
        # plt.plot(data[:100])
        # plt.show()
        return data

    def define_training_sample(self):
        # формируем обучающую выборку:
        data=self.build_sinusoid()

        # создаем обучающую выборку.Мы берем 3 отчета до и 3 отчета после:
        TRAINING_SAMPLE = np.array([np.diag(np.hstack((data[i:i+self.DIAPAZON],
                                                       data[i+self.DIAPAZON+1:i+self.TOTAL_LENGTH])))
                                    for i in range(self.COUNT_DOTS-self.TOTAL_LENGTH)])

        #создаем выборку целевых значений: мы делаем смещение на 3 отчета и далее берем все подряд:
        TARGET_SAMPLE=data[self.DIAPAZON:self.COUNT_DOTS-self.DIAPAZON-1]

        # print(TRAINING_SAMPLE,TARGET_SAMPLE, sep='\n')
        return TRAINING_SAMPLE, TARGET_SAMPLE, data

    def __init__(self):
        self.model=keras.Sequential([
            Input((self.TOTAL_LENGTH-1, self.TOTAL_LENGTH-1)),
            Bidirectional(GRU(2)),
            Dense(1, activation='linear')  #в задачах регресии всегда применяем такую ф-ю активации
        ])

        self.model.compile(optimizer=Adam(0.01), loss='mean_squared_error')

    def learn_RNN(self):
        TRAINING_SAMPLE, TARGET_SAMPLE, data=self.define_training_sample()
        history=self.model.fit(TRAINING_SAMPLE,TARGET_SAMPLE,batch_size=32,epochs=10)

    def PREDICTION(self):
        '''Делаем прогноз НС'''

        data = self.define_training_sample()[2]

        #определяем, сколько прогнозов мы будем делать:
        count_predicts=200

        #создаем коллекцию, в которую будут записываться наши прогнозы:
        PREDICTS=np.zeros(count_predicts)

        #копируем первые три элемента из коллекции data:
        PREDICTS[:self.DIAPAZON]=data[:self.DIAPAZON]

        # далее с помощью цикла мы формируем входные данные для НС:
        # строим диагональную матрицу, которая формируется из предыдущих элементов
        # PREDICTS и последующих элементов data:
        for i in range(count_predicts-self.DIAPAZON-1):
            input_data=np.diag(np.hstack((PREDICTS[i:i+self.DIAPAZON],
                                      data[i+self.DIAPAZON+1:i+self.TOTAL_LENGTH])))
            #добавляем нулевую ось,чтобы входной тензор соответствовал модели НС:
            input_data=np.expand_dims(input_data,axis=0)
            #получаем на выходе значение result
            result=self.model.predict(input_data)
            #это вычисленное значение добавляем в коллекцию PREDICTS:
            PREDICTS[i+self.DIAPAZON+1]=result

        #выводим результат:
        plt.plot(PREDICTS[:count_predicts]) #то, что мы спрогнозировали
        plt.plot(data[:count_predicts])  #то,что было
        plt.show()

user_Andrey=PredictDot()
user_Andrey.learn_RNN()
user_Andrey.PREDICTION()