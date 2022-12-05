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


class WordsPredict:
    '''RNN для прогнозирования следующего слова в тексте.
    Для обучающей выборки мы из текста будем выделять слова целиком,
    а набор уникальных слов будет составлять наш словарь, размер которого
    будет определяться переменной max_words_count_in_dict, затем каждое слово будет
    кодироваться OneHot-вектором в соответствии с его номером в словаре,
    переменная inp_words будет содержать количество слов на основе которых будет
    строиться прогноз следующего слова'''

    def load_text(self):
        '''Метод для загрузки текста из ПК'''
        with open('/home/andrey/Machine_Learning/ML_practice/datasets/text_for_rnn/texsts_samples.txt',
                  'r', encoding='utf-8') as my_text:
            text = my_text.read()
            text = text.replace('\ufeff', '')  # убираем первый невидимый символ
        return text

    def prepearing_data(self):
        '''Метод разбивки текста на отдельные слова с помощью Tokenizer,
        указывая, что у нас будет 20000 наиболее часто встречающихся слов в тексте, а остальные
        будут просто отброены сетью и она наних не будет обучаться,
        парметр filters удаляет все лишние символы из нашего текста,
        lower переврдит текст в нижний регистр, split - мы будем разбивать слова по пробелу,
        char_level=False потому что будем разбивать текст по словам, а не по символам.'''

        # указываем сколько максимум слов может быть у нас в словаре:
        max_words_count_in_dict = 2000

        # создаем токенайзер
        tokenizer = Tokenizer(num_words=max_words_count_in_dict,
                              filters='!-"-#$%amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r',
                              lower=True, split=' ', char_level=False)

        # далее пропускаем наш текст через токенайзер (текст берем из метода load_data),
        # чтобы придать каждому слову свое число
        tokenizer.fit_on_texts(self.load_text())

        # просмотр того, что у нас получается (для примера):
        # my_dict = list(tokenizer.word_counts.items())
        # print(my_dict[:10])

        # далее мы преобразовываем текст в последовательность чисел в соответствии
        # с полученным словарем, т.е. мы берем каждое отдельное слова в тексте
        # и на место этого слова ставим то число(индекс), которое соответствует этому слову:
        data = tokenizer.texts_to_sequences([self.load_text()])

        # далее преобразуем эту последовательность в OneHotEncoding-векторы (0 и 1):
        # result = to_categorical(data[0], num_classes=max_words_count_in_dict)
        result=np.array(data[0])

        # далее на основе коллекции result мы формируем 3-мерный тензор,
        # который у нас должен быть в обучающей выборке
        # Мы будем брать первые 3 слова и далее прогнозировать следующее слово,
        # потом мы смещаемся на 1 элемент вперед и повторяем операцию.

        input_words = 3
        n = result.shape[0] - input_words  # т.к. мы прогнозируем по трем словам четвертое

        # создаем тренировочную выборку:
        train = np.array([result[i:i + input_words] for i in range(n)])
        # строим целевую выборку:
        target = to_categorical(result[input_words:], num_classes=max_words_count_in_dict)

        return train, target, input_words, n, max_words_count_in_dict, tokenizer

    def __init__(self):
        train, target, input_words, n, max_words_count_in_dict, tokenizer = self.prepearing_data()

        self.model = keras.Sequential([
            Embedding(max_words_count_in_dict,512,input_length=input_words),
            SimpleRNN(256, activation='tanh'),
            Dense(max_words_count_in_dict, activation='softmax')
        ])

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.history=self.model.fit(train, target, batch_size=32, epochs=100, validation_split=0.2)

    def CreateText(self, text, len_text=10):
        '''Метод для составления текста'''
        # текст пользователя, который он вводит
        result = text
        tokenizer = self.prepearing_data()[5]
        input_words = self.prepearing_data()[2]
        max_words_in_dict = self.prepearing_data()[4]

        # преобразовываем слова в последовательность чисел в тексте
        data = tokenizer.texts_to_sequences([text])[0]

        for i in range(len_text):
            # формируем слова на основе которых делаем прогноз следующего слова:
            # преобразовываем коллекцию data  в векторы OneHotEncoding с 0 и 1:
            # OHE_vectors = to_categorical(data[i:i + input_words], num_classes=max_words_in_dict)

            # # создаем формат коллекции, которая подходит для подачи на RNN
            # collection = OHE_vectors.reshape(1, input_words, max_words_in_dict)

            #на вход НС будем подавать тензор из цифр
            digits = data[i:i + input_words]
            collection = np.expand_dims(digits, axis=0)


            # затем пропускаем эту коллекцию слов через нашу обученную модель:
            predict_words = self.model.predict(collection)

            # выбираем индекс с максимальным значением из коллекции слов predict_words:
            get_index = predict_words.argmax(axis=1)[0]

            # добавляем это слово в последовательность слов в тексте data
            data.append(get_index)

            # далее преобразовываем индекс обратно в слово и добавляем к тексту пользователя:
            result += ' ' + tokenizer.index_word[get_index]

        return result

    def MakePhrase(self,user_text,tex_length=10):
        result=user_text
        tokenizer = self.prepearing_data()[5]
        input_words = self.prepearing_data()[2]
        max_words_in_dict = self.prepearing_data()[4]

        data=tokenizer.texts_to_sequences([user_text])[0]
        for i in range(tex_length):
            # x=to_categorical(data[i:i+input_words],num_classes=max_words_in_dict)
            # inp=x.reshape(1,input_words,max_words_in_dict)
            #создаем список уже из индексов, а не из OHE-векторов
            digs_list=data[i:i+input_words]
            #добавляем ось
            input_collection=np.expand_dims(digs_list, axis=0)

            #делаем предсказание по обученной модели
            prediction=self.model.predict(input_collection)
            #ыбираем максимальное значение
            index=prediction.argmax(axis=1)[0]
            #добавляем в текст пользователя
            data.append(index)
            result+=' '+tokenizer.index_word[index]
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




user = WordsPredict()
print(user.MakePhrase('я люблю виски'))
user.show_acc_loss_during_learn_graphics()