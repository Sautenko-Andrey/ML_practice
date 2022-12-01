import tarfile
import urllib.request

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import os
from zlib import crc32
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

'''в рассматриваемом проекте  загрузим единственный сжатый файл housing.tgz ,
внутри которого содержится файл в формате разделенных запятыми значений
(comma separated value —- CSV) по имени housing.csv со всеми данными.'''


class HousesValue:
    '''Построение цен на жилье в США'''

    DOWONLOAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml2/master/'
    HOUSING_PATH = os.path.join('datasets', 'housing')
    HOUSING_URL = DOWONLOAD_ROOT + "datasets/housing/housing.tgz"

    def fetch_housing_data(self, housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
        '''Функция для извлечения данных.
        вызов fetch_housing_data() приводит к созданию каталога
datasets/housing в рабочей области, загрузке файла housing.tgz , извле-
чению из него файла housing, csv и его помещению в указанный каталог.'''
        os.makedirs(housing_path, exist_ok=True)
        tgz_path = os.path.join(housing_path, 'housing.tgz')
        urllib.request.urlretrieve(housing_url, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=housing_path)
        housing_tgz.close()

    def load_housing_data(self, housing_path=HOUSING_PATH):
        '''Загрузим данные с приминением pandas.
        Эта функция возвращает pandas-объект DataFrame,
        который содержит все данные по домам'''

        csv_path = os.path.join(housing_path, 'housing.csv')
        return pd.read_csv(csv_path)

    def show_brief_data_description(self):
        '''Функция выводит в консоль структуру таблицы данных по домам и
        краткое описание данных, также какие есть категории у столбца <ocean_proximety>
        и сколько округов принадлежит к каждой из категорий,
        сводку по числовым атрибутам (остальные поля)'''
        housing = self.load_housing_data()
        print(f'Верхние 5 строк набора данных\n{housing.head()}\n----------------------------')
        print(f'Краткое описание данных:\n{housing.info()}\n-----------------------')
        print('Категории по близости к океану и сколько округов принадлежит к каждой категории:')
        print(f'{housing["ocean_proximity"].value_counts()}\n-------------------------------')
        print(f'Сводка по числовым атрибутам:\n{housing.describe()}\n----------------------')

    def show_gistograms(self):
        '''Функция строит гистограммы для каждого числового атрибута.
        Гистограмма показывает количество образцов по вертикальной оси,
        которые имеют заданный диапазон значений по горизонтальной оси.'''

        housing = self.load_housing_data()
        housing.hist(bins=10, figsize=(3, 3))
        plt.show()

    # РУЧНОЕ РАЗДЕЛЕНИЕ ДАННЫХ
    def split_train_test(self, data, test_ratio):
        '''Функция для отделения испытатльного набора из обучающей выборки (апасной не лучший вариант)'''
        # перемешаем данные случайным образом и занесем их в массив Numpy
        shuffled_indices = np.random.permutation(len(data))
        # определим размер тестовой выборки (возьмем нужный % от всего датасета)
        test_set_size = int(len(data) * test_ratio)
        # возьмем данные для теста
        test_indices = shuffled_indices[:test_set_size]
        # возьмем данные для обучения
        train_indices = shuffled_indices[test_set_size:]
        # вернем 2 объекта Pandas
        return data.iloc[train_indices], data.iloc[test_indices]
        # будем использовать train_set,test_set=split_train_test(housing,0.2)

    # более правильный подход к выделению данных для теста. Мы вычисляем хеш-значение идентификатора
    # каждого образца и помещаем образец в испытательный набор, если его значение <=20% максимального
    # хеш-значения.
    # Такой подход гарантирует,
    # что испытательный набор будет сохраняться согласованным между множеством запусков программы,
    # даже если набор данных обновляется.Новый испытательный набор будет содержать 20% новых образцов,
    # но не включать образцы,которые присутствовали в испытательном наборее ранее.
    # для этого используются ф-ии test_set_check(), split_train_test_by_id()

    def test_set_check(self, identifer, test_ratio):
        '''Функция проверки на то,что хеш-значение идентификатора каждого образца
        меньше или равно 20 % максимального максимального хеш-значения.
        Т.к. мы берем 20% для тестовой выборки из тренировочной выборки'''

        return crc32(np.int64(identifer)) & 0xffffffff < test_ratio * 2 ** 32

    def split_traint_test_by_id(self, data, test_ratio, id_column):
        ids = data[id_column]
        in_test_set = ids.apply(lambda id_: self.test_set_check(id, test_ratio))
        return data.loc[~in_test_set], data.loc[in_test_set]

    # К сожалению, набор данных housing не имеет столбца для идентифика-
    # тора. Простейшее решение предусматривает применение в качестве иденти-
    # фикатора индекса строки:
    def get_train_and_test_sets(self):
        '''Если мы используем индекс строки как уникальный идентификатор,
                то мы должны удостовериться в том,что новые данные добавляются в конец набора данных
                и никакие строки из набора не удаляются.
                Если невозможно добиться этого контроля,то мы попробуем здесь
                применять для построения уникального идентификатора самые стабильные признаки из
                нашего набора данных, т.е. широта и долгота округа. Мы скомбинируем широту и долготу
                в идентификатор'''
        housing = self.load_housing_data()
        # добавляем столбец index  в наш датасет для идентификации
        housing_with_id = housing.reset_index()

        # получаем окончательную тренировочную и тестовую выборки ВАРИАНТ №1
        # train_set, test_set=self.split_traint_test_by_id(housing_with_id,0.2,'index')

        # получаем окончательную тренировочную и тестовую выборки
        # АЛЬТЕРНАТИВНЫЙ ВАРИАНТ ЛУЧШИЙ В ДАННОМ СЛУЧАЕ!!!
        housing_with_id['id'] = housing['longitude'] * 1000 + housing['latitude']
        train_set, test_set = self.split_traint_test_by_id(housing_with_id, 0.2, 'id')
        return train_set, test_set

    # РАЗДЕЛЕНИЕ ДАННЫХ С ПОМОЩЬЮ SCIKIT_LEARN!
    def get_train_and_test_sets_ALTERNATIVE(self):
        '''Альтернативное разделение набора данных
         на тестовую и тренировочную выборки с помощью библиотеки Scikit-Learn
         и ее ф-ии train_test_split()
         random_state=42 - устанавливает начальное значение генератора случайных чисел
         (так делается всегда и 42-здесь случайное число)'''
        housing = self.load_housing_data()
        train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    # ТРЕТИЙ И АКТУАЛЬНЫЙ ДЛЯ ЭТОЙ ЗАДАЧИ ВАРИАНТ РАЗДЕЛЕНИЯ ВЫБОРКИ!!!
    def get_stratific_sample(self):
        '''Сделаем 5 страт на основании среднего дохода (разобъем все на категории дохода)
        тем самым мы обеспечим репрезетативность испытательного набора
        для различных категорий дохода в целом наборе данных'''

        housing = self.load_housing_data()
        housing['income_cat'] = pd.cut(housing['median_income'],
                                       bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                       labels=[1, 2, 3, 4, 5])
        # теперь сделаем стратицифированную выборку на основе категории дохода.
        # Используем класс StratifiedShuffleSplit из Scikit-Learn:
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(housing, housing['income_cat']):
            strat_train_set = housing.loc[train_index]
            strat_test_set = housing.loc[test_index]

        # удалим атрибут income_cat ,чтобы вернуть данные в первоначальное состояние
        for set_ in (strat_train_set, strat_test_set):
            set_.drop('income_cat', axis=1, inplace=True)

        return strat_train_set, strat_test_set


user_1 = HousesValue()
user_1.show_brief_data_description()
user_1.show_gistograms()
