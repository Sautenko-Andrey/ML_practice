import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

import keras
from tensorflow import keras

from keras.layers import Dense, Embedding, GRU
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


class WhatAreYouTalkingAbout:
    '''НС , которая предсказывает тему разговора: о спорте,
    машинах или о женщинах.'''

    # задаем количество наиболее часто употребляемых слов в разговоре
    MAX_POP_WORDS = 1000

    MAX_LEN_TEXT = 12

    def __init__(self):
        self.model = keras.Sequential([
            Embedding(self.MAX_POP_WORDS, 128, input_length=self.MAX_LEN_TEXT),
            GRU(128, return_sequences=True),
            GRU(64),
            Dense(3, activation='softmax')
        ])

        self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                           optimizer=Adam(0.0001))

    def training_RNN(self):
        TRAIN_DATA, TARGET_DATA, tokenizer = self.prepearing_dataset()
        history = self.model.fit(TRAIN_DATA, TARGET_DATA, batch_size=32, epochs=50)

        reverse_word_map = dict(map(reversed, self.prepearing_dataset()[2].word_index.items()))

        return history, reverse_word_map

    def uload_data(self):
        # загружаем разговоры о машинах:
        with open('/home/andrey/Machine_Learning/нейронные_сети/SPORT_CARS_GIRLS_MESSAGE/cars_talks.txt', 'r',
                  encoding='utf-8') as f:
            cars_talks = f.readlines()

        # загружаем разговоры о девушках:
        with open('/home/andrey/Machine_Learning/нейронные_сети/SPORT_CARS_GIRLS_MESSAGE/girls_talk.txt', 'r',
                  encoding='utf-8') as f:
            girls_talks = f.readlines()

        # загружаем разговоры о спорте:
        with open('/home/andrey/Machine_Learning/нейронные_сети/SPORT_CARS_GIRLS_MESSAGE/sport_talls.txt', 'r',
                  encoding='utf-8') as f:
            sports_talks = f.readlines()

        # объединяем все разговоры вместе:
        all_talks = cars_talks + girls_talks + sports_talks

        # считаем количество разговоров о машинах:
        count_cars_talks = len(cars_talks)
        # считаем количество разговоров о девушках:
        count_girls_talk = len(girls_talks)
        # считаем разговоры о спорте:
        count_sports_talks = len(sports_talks)

        return all_talks, count_cars_talks, count_girls_talk, count_sports_talks

    def prepearing_dataset(self):
        '''Подготовка данных перед обучением НС'''

        # загружаем объединенный набор разговоров и размеры каждлого из них по отдельности:
        all_talks, count_cars_talks, count_girls_talk, count_sports_talks = self.uload_data()

        # создаем токенайзер:
        tokenizer = Tokenizer(num_words=self.MAX_POP_WORDS,
                              filters='!-"-#$%amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r',
                              lower=True, split=' ', char_level=False)

        # пропускаем текст через токенайзер:
        tokenizer.fit_on_texts(all_talks)

        # формируем числовую последовательность из слов в тексте:
        data = tokenizer.texts_to_sequences(all_talks)

        # длинные отзывы обрезаем на 1-м слове, а коротки е дополняем нулями:
        data_pad = pad_sequences(data, maxlen=self.MAX_LEN_TEXT)

        # формируем тренировочную и целевую выборки:
        TRAIN_SAMPLE = data_pad
        TARGET_SAMPLE = np.array(
            [[1, 0, 0]] * count_cars_talks + [[0, 1, 0]] * count_girls_talk + [[0, 0, 1]] * count_sports_talks)

        # перемешаем обучающую выборку для более качественной тренировки:
        indeces = np.random.choice(TRAIN_SAMPLE.shape[0], size=TRAIN_SAMPLE.shape[0],
                                   replace=False)

        TRAIN_SAMPLE = TRAIN_SAMPLE[indeces]
        TARGET_SAMPLE = TARGET_SAMPLE[indeces]

        return TRAIN_SAMPLE, TARGET_SAMPLE, tokenizer

    def index_convert_to_text(self, indeces_list):
        '''Метод для преобразования индексов в слова'''
        reverse_word_map = self.training_RNN()[1]
        normal_text = [reverse_word_map.get(x) for x in indeces_list]
        return (normal_text)

    def get_predict(self, user_text):
        '''Метод для получения прогноза'''
        # переводим отзыв пользователя в нижний регистр
        user_text = user_text.lower()

        # далее пропускаем отзыв через токенайзер и преобразуем слова в числа:
        tokenizer = self.prepearing_dataset()[2]
        data = tokenizer.texts_to_sequences([user_text])

        # преобразовываем в вектор нужной нам длины как и у обучающих данных:
        data_pad = pad_sequences(data, maxlen=self.MAX_LEN_TEXT)

        # смотрим какую на самом деле фразу мы анализируем(т.к. некоторых слов у нас может не быть в словаре)
        print(self.index_convert_to_text(data[0]))

        # получаем прогноз:

        result = self.model.predict(data_pad)
        print(result, np.argmax(result), sep='\n')
        if np.argmax(result) == 0:
            print('Разговор о машинах.')
        elif np.argmax(result) == 1:
            print('Разговор о девушках.')
        elif np.argmax(result) ==2:
            print('Разговор о спорте.')
        else:
            print('Ошибка. Я не могу определить что это за разговор...')
