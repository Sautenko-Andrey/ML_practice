import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import re
import matplotlib.pyplot as plt

import keras
from tensorflow import keras

from keras.layers import Dense, SimpleRNN, Input
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer


class SymbolPredict:
    '''РНН на вход которой подаются символы,
     а на выходе она будет строить прогноз следующего сивола
     Мы будем последовательно подавать 6 символов и на выходе
      будем ожидать прогноз следующего символа (ахитектура сети - Many to One)
      Кодироватьсимволы будем с помощью OneHotEncoding'''

    def upload_text_file(self):
        '''Загрузка файла с текстом.
        Убираем первый невидимый символ.
        Отсавляем только символы русского алфавита'''

        with open('/home/andrey/Machine_Learning/ML_practice/datasets/text_for_rnn/texsts_samples.txt',
                  'r', encoding='utf-8') as f:
            text = f.read()
            text = text.replace('\ufeff', '')  # убираем первый невидимый символ
            text = re.sub(r'[^А-я ]', '', text)  # заменяем все символы кроме кириллицы на пустые символы
        return text

    def text_parsing(self):
        '''Подготовка текста'''
        text = self.upload_text_file()  # создаем ссылку на уже загруженный текст
        total_symbols = 34  # всего будет 34 символа= 33 буква алфавита + пробел

        # создаем объект класса Tokenizer, на выходе он даст разбивку текста на соответствующие элементы
        tokenizer = Tokenizer(num_words=total_symbols,
                              char_level=True)  # токенизируем на уровне символов(char_level=True)

        # с помощью метода fit_on_text пропускаем текст через парсер tokenizer
        # и получаем результат преобразования
        tokenizer.fit_on_texts([text])  # формируем токены на основе частотности в тексте

        # покажем результат - коллекция word_index
        # print(tokenizer.word_index) # каждому символу назначется определенное число
        return tokenizer, total_symbols

    def convert_to_ONEHOTENCODING(self):
        '''Преобразование всего текста в набор векторов с 0 и 1
        (единица стоит в оответсвующем индексе)
        c помощью texts_to_matrix. И на выходе мы будем получать
         коллекцию data, которая имеет нужный формат'''

        # загружаем текст
        text = self.upload_text_file()  # создаем ссылку на уже загруженный текст

        # загружаем парсер
        tokenizer = self.text_parsing()[0]

        inp_chars = 6  # кол-во символов, которые мы будем подавать на вход
        data = tokenizer.texts_to_matrix(text)  # преобразовываем текст в массив OneHotEncoding векторов
        # print(data.shape)
        # print(data)
        n = data.shape[0] - inp_chars  # т.к. мы предсказываем по 6 символам четвертый символ
        return data, n, inp_chars

    def __init__(self):
        '''Инициализируем тренировочеую и целевую выборки,
        сразу прописываем архитектуру RNN, компилируем ее и тренируем.'''
        data, n, inp_chars = self.convert_to_ONEHOTENCODING()
        total_symbols = self.text_parsing()[1]

        # делаем обучающую выборку

        # формирование входного тензора
        self.train = np.array([data[i:i + inp_chars, :] for i in range(n)])

        # требуемые выходы НС
        self.target = data[inp_chars:]

        # формируем тестовую выборку

        # формируем модель RNN
        self.model = keras.Sequential([
            Input((inp_chars, total_symbols)),  # при тренировке в RNN keras подается сразу вся последовательность
            SimpleRNN(256, activation='tanh'),  # рекурентный слой
            Dense(total_symbols, activation='softmax')
        ])

        # компилируем сеть
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam', metrics=['accuracy'])

        # тренируем НС
        self.history = self.model.fit(self.train, self.target, batch_size=32, epochs=50, validation_split=0.2)

    def show_model_architecture(self):
        print('Архитерктура RNN')
        print(self.model.summary())

    # def model_value(self):
    #     test_loss, test_acc = self.model.evaluate(self.x_test,self.y_test)
    #     print("Точность и потери модели на тестовых изображениях")
    #     print('---------------------------------------------------')
    #     print(f'Точость модели: {test_acc}')
    #     print(f'Потери модели: {test_loss}')

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

    def build_phrase(self, input_symbols, str_len=50):
        '''Вспомагательная функия для построения фраз
        на основе прогнозных значений.
        На ее вход мы передаем начальные символы inp_str, по которым она будет строить прогноз'''

        inp_chars = self.convert_to_ONEHOTENCODING()[2]
        tokenizer = self.text_parsing()[0]
        total_symbols = self.text_parsing()[1]

        for i in range(str_len):
            x = []
            for j in range(i, i + inp_chars):
                x.append(tokenizer.texts_to_matrix(input_symbols[j]))  # реобразование символов в OneHotEncoding
            x = np.array(x)

            # формируем начальные символы, по которым будем делать прогноз в формате OHE
            inp = x.reshape(1, inp_chars, total_symbols)

            # подаем эти символы на вход НС для предсказания символа(получим вектор из 34 эл-в)
            pred = self.model.predict(inp)

            # далее из этого вектора выбираем символ, который имеет наибольшое значение(индекс наибольшего значения)
            symbol = tokenizer.index_word[pred.argmax(axis=1)[0]]

            # добавляем символ к строке
            input_symbols += symbol
        return input_symbols
