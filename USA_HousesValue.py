import tarfile
import urllib.request

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import os
from zlib import crc32
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from pandas.plotting import scatter_matrix

'''в рассматриваемом проекте  загрузим единственный сжатый файл housing.tgz ,
внутри которого содержится файл в формате разделенных запятыми значений
(comma separated value —- CSV) по имени housing.csv со всеми данными.'''


class HousesValue:
    '''Построение цен на жилье в США'''

    DOWONLOAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml2/master/'
    HOUSING_PATH = os.path.join('datasets', 'housing')
    HOUSING_URL = DOWONLOAD_ROOT + "datasets/housing/housing.tgz"

    def __init__(self):
        # инициализируем обучающую и тестовую выборки данных
        self.strat_train_set = self.get_stratific_sample()[0]
        self.strat_test_set = self.get_stratific_sample()[1]

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

    # ОБНАРУЖЕНИЕ И ВИЗУАЛИЗАЦИЯ ДАННЫХ ДЛЯ ПОНИМАНИЯ ИХ СУЩНОСТИ
    def work_with_train_data_copy(self):
        # создадим копию обучающей выборки,чтобы не навредить оригиналу
        housing = self.strat_train_set.copy()

        # поскольку имеется географическая информация (широта и долгота),
        # создадим график рассеяния всех округов для визуализации данных
        # где мы будем четко видеть области с высокой плотностью - область залива Сан Франциско
        # и поблизости Лос-Анжелеса и Сан-Диего плюс длинная линия довольно высокой плот-
        # ности в Большой Калифорнийской долине, в частности около Сакраменто и
        # Фресно.
        housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)
        plt.show()

        # далее обратимся к ценам на дома и визуализируем их
        # Радиус каждого круга представляет население округа (параметр s), а
        # цвет — цену (параметр с). Мы будем применять предварительно определен-
        # ную карту цветов (параметр сmар) по имени jet, которая простирается от
        # синего цвета (низкие цены) до красного (высокие цены)
        housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
                     s=housing['population'] / 100, label='population', figsize=(10, 7),
                     c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True, )
        plt.legend()
        plt.show()

        # ПОИСК СВЯЗЕЙ
        # вычислим стандартный коэф.корреляции между каждой парой атрибутов
        corr_matrix = housing.corr()
        # далее посмотрим на сколько каждый атрибут связан со средней стоимостью дома:
        print(corr_matrix['median_house_value'].sort_values(ascending=False))

    def correlation_amount_attributes(self):
        '''Альтернативный способ проверки кореляции между атрибутами
        с использованием pandas-функции scatter_matrix() ,
        котоая вычерчивает каждый числовой атрибут
         по отношению к другому числовому атрибуту.
         Мы сосредоточим внимание не на всех атрибутах,а на наиболее многообещающих,
         которые представляются наиболее связанными со средней стоимостью дома.'''
        # создадим копию обучающей выборки,чтобы не навредить оригиналу
        housing = self.strat_train_set.copy()
        attributes = ['median_house_value', 'median_income', 'total_rooms',
                      'housing_median_age']  # эти атрибуты наиболее значимые для средней стоимости дома
        scatter_matrix(housing[attributes], figsize=(12, 8))
        plt.show()

        # Самым многообещающим атрибутом для прогнозирования средней
        # стоимости дома является медианный доход,
        # поэтому давайте увеличим его график рассеяния корреляции
        housing.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.1)
        plt.show()

    # экспериментирование с комбинациями атрибутов
    def experimenting_with_attributes_combinations(self):
        housing = self.strat_train_set.copy()

        # создадим атрибут количество комнат на дом
        housing['rooms_per_household'] = housing['total_rooms'] / housing['households']

        # создадим атрибут количество спален на комнаты
        housing['bedrooms_per_room'] = housing['total_bedrooms'] / housing['total_rooms']

        # создадим атрибут население на дом
        housing['population_per_household'] = housing['population'] / housing['households']

        # снова создадим матрицу корреляции
        corr_matrix = housing.corr()
        print(corr_matrix['median_house_value'].sort_values(ascending=False))

        # ВЫВОД ПО ЭТОМУ ПРИНТУ!!!!!!!:
        # Новый атрибут bedrooms_per_room (количество спален
        # на количество комнат) намного больше связан со средней стоимостью дома,
        # чем общее количество комнат или спален. По всей видимости, дома с мень-
        # шим соотношением спальни/комнаты имеют тенденцию быть более доро-
        # гостоящими. Количество комнат на дом также более информативно, нежели
        # суммарное число комнат в округе — очевидно, чем крупнее дома, тем они
        # дороже.

    # ПОДГОТОВКА ДАННЫХ ДЛЯ АЛГОРИТМОВ МАШИННОГО ОБУЧЕНИЯ
    def prepearing_data_for_ML_alg(self):
        ''''''
        # скопируем обучающий набо и разделим на прогнозаторы и метки
        housing = self.strat_train_set.drop('median_house_value', axis=1)
        housing_labels = self.strat_train_set['median_house_value'].copy()
        return housing, housing_labels

    # ОЧИСТКА ДАННЫХ
    def clear_data(self):
        '''Большинство алгоритмов ML не могут работать с недостающими признаками,
    поэтому мы создадим несколько функций, которые позаботяться об этом.
    Мы видели, что в атрибуте total_bedrooms не хватало ряда значений
    ,поэтому мы исправим это положение. Есть 3 варианта:
    1) избавиться от соответствующих округов
    2) избавиться от всего атрибута
    3) установить недостающие значения в некоторую величину (ноль, среднее, медиана и т.д.).'''

        # загрузим подготовленные данные:
        housing, housing_labels = self.prepearing_data_for_ML_alg()

        # все эти варианты легко реализуются с помощью методов dropna(), drop(), filna():
        # вариант №1 - избавиться от соответствующих округов
        # housing.dropna(subset=['total_bedrooms'])

        # вариант №2 - избавиться от всего атрибута
        # housing.drop('total_bedrooms',axis=1)

        # вариант №3 - установить недостающие значения в некую величину
        # housing['total_bedrooms'].fillna(median,inplace=True)

        # библиотека Scikit-Learn  предлагает удобный класс,
        # который заботиться об отсутствующих значениях SimpleImputer

        # создадим экземпляр класса SimpleInputer, указав на то, что недостающие значения
        # каждого атрибута следует заменить медианой этого атрибута
        imputer = SimpleImputer(strategy='median')

        # т.к. медиана подсчитывается только на числовых атрибутах,
        # нам требуется создать копию данных без текстового атрибута
        # ocean_proximity:
        housing_num = housing.drop('ocean_proximity', axis=1)

        # теперь экземпляр imputer можно подогнать к обучающим данным с приминением метода fit():
        imputer.fit(housing_num)

        # проверяем и сравниваем:
        print(imputer.statistics_)
        print(housing_num.median().values)

        # если все нормально, теперь мы можем использовать "обученный" экземпляр imputer
        # для трансформации обучающего набора путем замены недостающих значений известными медианами
        X = imputer.transform(housing_num)

        # результатом является обыкновенный массив Numpy , содержащий трансформированные признаки.
        # если его нужно поместить обратно в pandas-лбъект,то:
        housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
        return X, housing_tr

    # ОБРАБОТКА ТЕКСТОВЫХ И КАТЕГОРИАЛЬНЫХ АТРИБУТОВ
    def handling_text_and_categorical_attrs(self):
        '''В нашем наборе есть всего один текстовый атрибут ocean_proximity'''

        # загрузим подготовленные данные:
        housing, housing_labels = self.prepearing_data_for_ML_alg()

        # ассмотрим значения атрибута ocean_proximity для первых 10 образцов
        housing_cat = housing[['ocean_proximity']]
        print(housing_cat.head(10))

        # мы видим, что это не произвольный текст и атрибут ocean_proximity является категориальным
        # т.е. алгоритмы ML любят работать с числами, то мы преобразуем категории из текста в числа,
        # применив класс OrdinalEncoder из Scikit-Learn:
        ordinal_encoder = OrdinalEncoder()
        housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
        # посмотрим что получилось:
        print(housing_cat_encoded[:10])

        # так же просмотрим список категорий:
        print(ordinal_encoder.categories_)

        return housing_cat_encoded

    # но предыдущий метод нам не особо подходит, поэтому сделаем другой
    def handling_text_and_categorical_attrs_WITH_1HOTENCODER(self):
        '''В нашем наборе есть всего один текстовый атрибут ocean_proximity.
        Предыдущий метод handling_text_and_categorical_attrs нам не подходит,
        класс OneHotEncoder для преобразования категориальных значений в векторы в унитарном коде'''

        # загрузим подготовленные данные:
        housing, housing_labels = self.prepearing_data_for_ML_alg()
        housing_cat = housing[['ocean_proximity']]
        cat_encoder = OneHotEncoder()
        housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
        return housing_cat_1hot

    # МАСШТАБИРОВАНИЕ ПРИЗНАКОВ

    #Конвееры трансформации
    
    


user_1 = HousesValue()
# user_1.work_with_train_data_copy()
# user_1.correlation_amount_attributes()
# user_1.experimenting_with_attributes_combinations()
# user_1.prepearing_data_for_ML_alg()
# user_1.clear_data()
user_1.handling_text_and_categorical_attrs()
