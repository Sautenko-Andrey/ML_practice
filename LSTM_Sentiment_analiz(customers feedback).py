import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

import keras
from tensorflow import keras

from keras.layers import Dense, Embedding, LSTM
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences


class GuessFeedback:
    # задаем количество наиболее часто встречаемых слов в отзывах
    MAX_WORDS = 1000

    # определяем количество слов,к которому будет приведен каждый отзыв:
    MAX_LEN_TEXT = 10

    def __init__(self):
        self.model = keras.Sequential([
            Embedding(self.MAX_WORDS, 128, input_length=self.MAX_LEN_TEXT),
            LSTM(128, return_sequences=True),
            LSTM(64),
            Dense(2, activation='softmax')
        ])

        self.model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                           optimizer=Adam(0.0001))

    def training_RNN(self):
        TRAIN_DATA, TARGET_DATA, tokenizer = self.converted_data()
        history = self.model.fit(TRAIN_DATA, TARGET_DATA, batch_size=32, epochs=50)

        reverse_word_map = dict(map(reversed, self.converted_data()[2].word_index.items()))

        return history, reverse_word_map

    def upload_data(self):
        # загружаем положительные отзывы:
        with open('/home/andrey/Machine_Learning/нейронные_сети/ALL_PHRASES/positive_text.txt', 'r',
                  encoding='utf-8') as f:
            positive_texts = f.readlines()
            # убираем первый невидимый символ
            positive_texts[0] = positive_texts[0].replace('\ufeff', '')

        # загружаем отрицательные отзывы:
        with open('/home/andrey/Machine_Learning/нейронные_сети/ALL_PHRASES/negative_text.txt', 'r',
                  encoding='utf-8') as f:
            negative_texts = f.readlines()
            # убираем первый невидимый символ
            negative_texts[0] = positive_texts[0].replace('\ufeff', '')

        # объединяем положительные и негативные отзывы:
        texts = positive_texts + negative_texts
        # подсчитаем количество положительных отзывов:
        count_positive_texts = len(positive_texts)
        # подсчитаем количество отрицательных отзывов:
        count_negative_texts = len(negative_texts)

        return texts, count_positive_texts, count_negative_texts

    def converted_data(self):
        # загружаем текст для обработки:
        texts = self.upload_data()[0]
        # прописываем токенайзер:
        tokenizer = Tokenizer(num_words=self.MAX_WORDS,
                              filters='!-"-#$%amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r',
                              lower=True, split=' ', char_level=False)
        # прогоняем текст через токенайзер:
        tokenizer.fit_on_texts(texts)
        # формируем последовательность из чисел вместо слов (коллекция индексов)
        data = tokenizer.texts_to_sequences(texts)
        # короткие отзывы дополняем нулями, длинные обрезаем до 10 слов в отзыве:
        data_pad = pad_sequences(data, maxlen=self.MAX_LEN_TEXT)
        # окончательно формируем обучающую выборку:
        TRAIN_SAMPLE = data_pad
        TARGET_SAMPLE = np.array([[1, 0]] * self.upload_data()[1] + [[0, 1]] * self.upload_data()[2])
        # перемешаем обучающую выборку для лучшей тренированости НС:
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
        tokenizer = self.converted_data()[2]
        data = tokenizer.texts_to_sequences([user_text])

        # преобразовываем в вектор нужной нам длины как и у обучающих данных:
        data_pad = pad_sequences(data, maxlen=self.MAX_LEN_TEXT)

        # смотрим какую на самом деле фразу мы анализируем(т.к. некоторых слов у нас может не быть в словаре)
        print(self.index_convert_to_text(data[0]))

        # получаем прогноз. если перменная argmax принимает значение 0 ( 0 - это первый нейрон,
        # отвечающий за положительные отзывы),то отзыв положительный,
        # если 1 , то отрицательный

        result = self.model.predict(data_pad)
        print(result, np.argmax(result), sep='\n')
        if np.argmax(result) == 0:
            print('Положительный отзыв от турсита.')
        else:
            print('Отрицательный отзыв от туриста...')


user_Andrey = GuessFeedback()
user_Andrey.training_RNN()
user_Andrey.get_predict('Мы с женой очень довольны отдыхом. Спасибо команде анимации и всем барменам!')
