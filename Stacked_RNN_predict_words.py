import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import re
import matplotlib.pyplot as plt

import keras
from tensorflow import keras

from keras.layers import Dense, SimpleRNN, Input, Embedding
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils import to_categorical


class PredictWord_StackedRNN:
    '''Рекурентная нейронная сеть Stackd RNN,
    предсказывающая слово по трем предыдущим.
    Архитекутра представляет собой '''

    # загружаем текст:
    with open('/home/andrey/Machine_Learning/ML_practice/datasets/text_for_rnn/texsts_samples.txt',
              'r', encoding='utf-8') as f:
        TEXT = f.read()

        # убираем первый невидимый символ:
        TEXT = TEXT.replace('\ufeff', '')

        # задаем количество наиболее часто повторяющихся слов в тексте
        MAX_WORDS = 2000

        # кол-во слов по которым мы будем строить прогноз:
        INPUT_WORDS = 3

    def prepearing_text(self):
        # создаем токенайзер
        tokenizer = Tokenizer(num_words=self.MAX_WORDS,
                              filters='!-"-#$%amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r',
                              lower=True, split=' ', char_level=False)
        # пропускаем загруженный текст через токенайзер, назначая каждому слову свой индекс в словаре:
        tokenizer.fit_on_texts([self.TEXT])
        # создаем последовательность данных в формате индексов слов
        data = tokenizer.texts_to_sequences([self.TEXT])
        # переводим результат в массив Numpy
        result = np.array(data[0])
        # получаем количествотренировочных тензоров для обучения RNN:
        number = result.shape[0] - self.INPUT_WORDS
        # определяем тренировочную выборку
        Train = np.array([result[i:i + self.INPUT_WORDS] for i in range(number)])
        # определяем целевую выборку(т.е. прогнозы)
        Target = to_categorical(result[self.INPUT_WORDS:], num_classes=self.MAX_WORDS)
        return Train, Target, tokenizer

    def __init__(self):
        self.model = keras.Sequential([
            Embedding(self.MAX_WORDS, 512, input_length=self.INPUT_WORDS),
            SimpleRNN(256, activation='tanh', return_sequences=True),
            SimpleRNN(128, activation='tanh', return_sequences=True),  # работает ManyToMany
            # получим на выходе тензор (batch_size,timesteps,units)
            # batch_size - размер пакета данных,timesteps-размер последовательности в каждом примере(INPUT_WORDS)
            # units-число нейронов в RNN-слое. Такое должно подаваться навход каждому RNN-слою,
            # поэтому дополнительно прописываем return_sequences=True в первоем слое SimpleRnn
            SimpleRNN(64, activation='tanh'),  # работает ManyToOne
            Dense(self.MAX_WORDS, activation='softmax')
        ])

        self.model.compile(optimizer='adam', loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self.history = self.model.fit(self.prepearing_text()[0], self.prepearing_text()[1],
                                      epochs=50, batch_size=32, validation_split=0.2)

    def GetPhrase(self, user_text, text_length=10):
        result = user_text
        tokenizer = self.prepearing_text()[2]

        # пропускаем текст пользователя через токенайзер и получаем последовательность чисел:
        data = tokenizer.texts_to_sequences([user_text])[0]

        for i in range(text_length):
            # создаем список уже из индексов, а не из OHE-векторов
            digs_list = data[i:i + self.INPUT_WORDS]
            # добавляем ось
            input_collection = np.expand_dims(digs_list, axis=0)

            # делаем предсказание по обученной модели
            prediction = self.model.predict(input_collection)
            # ыбираем максимальное значение
            index = prediction.argmax(axis=1)[0]
            # добавляем в текст пользователя
            data.append(index)
            result += ' ' + tokenizer.index_word[index]
        return result

    def show_acc_loss_during_learn_graphics(self):
        '''Выведем графики точности и потерь при обучении RNN'''
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()
