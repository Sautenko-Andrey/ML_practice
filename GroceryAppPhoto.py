import os, shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt

import keras
from tensorflow import keras
from keras.layers import MaxPooling2D, Conv2D, Flatten, Dense, Dropout
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator


class GroceryAppPhoto:
    '''Сверточная нейронная сеть для распознования продуктов питания
    на фотографиях пользователя сайта.
    Я возьму сверточную НС, обученную на наборе ImageNet (1,4 млн изображений
     классифицированных на 1000 разных классов), так же воспользуюсь архитектурой
     VGG-16. Реализую выделение признаков , т.е. буду использовать представления, изученные
     предыдущей НС (VGG-16 на наборе ImageNet),для выделения признаков из новых образцов,
     которые затем будут пропускаться через совершенно новый классификатор, обучаемый с нуля под
     мои требования.Я буду использовать только несколько первых слоев предобученной НС.
    '''

    # получим доступ к папке с обучающей выборкой grocery_data
    BASE_DIR = '/home/andrey/grocery_data'
    # доуступ к папке train
    TRAIN_DIR = os.path.join(BASE_DIR, 'train')
    # доступ к папке validation
    VALIDATION_DIR = os.path.join(BASE_DIR, 'validation')
    # доступ к папке test
    TEST_DIR = os.path.join(BASE_DIR, 'test')

    COUNT_CHERNIGIVSKE_PICS = 131  # кол-во фоток пива в тренировочной выборке
    COUNT_HETMAN_PICS = 122  # количество фоток водки в тренировочной выборке

    # подсчитаем количество изображений в каждой из папок
    COUNT_TRAIN_DATA = 253
    COUNT_VALIDATION_DATA = 87
    COUNT_TEST_DATA = 100

    # Сначала запустим экземпляры класса ImageDataGenerator,
    # чтобы извлечь изображения и их метки в массивы Numpy
    DATAGEN = ImageDataGenerator(rescale=1. / 255)

    # Определяем количество батчей
    BATCH_SIZE = 20

    def prepearing_data(self):
        '''Создание генераторов, которые преобразовывают изображения в необходимые тензоры'''
        pass

    def __init__(self):
        '''Инициализация всех необходимых объектов'''

        # Создание экземпляра сверточной основы VGG-16
        self.conv_base = VGG16(
            weights='imagenet',  # указываем, что берем обученные весовые коэф. на базе ImageNet
            include_top=False,  # указывае, что не подключаем уже обученный классификатор (у нас булет свой с нуля)
            # input_shape=(150,150,3)  #размерность тензоров изображений, подающихся на вход НС (необязательный параметр)
            # не указывая его мы делаем так, что наша сеть может обрабатывать изображения любого размера
        )

        # инициализируем данные для обучения:
        train_features, train_labels = self.extract_features(self.TRAIN_DIR, self.COUNT_TRAIN_DATA)
        validation_features, validation_labels = self.extract_features(self.VALIDATION_DIR, self.COUNT_VALIDATION_DATA)
        test_features, test_labels = self.extract_features(self.TEST_DIR, self.COUNT_TEST_DATA)

        # В настоящий момент выделенные признаки имеют форму (образцы, 4, 4, 512). Мы
        # будем передавать их на вход полносвязного классификатора, поэтому мы должны
        # привести этот тензор к форме (образцы, 8192):
        train_features = np.reshape(train_features, (self.COUNT_TRAIN_DATA, 4 * 4 * 512))
        validation_features = np.reshape(validation_features, (self.COUNT_VALIDATION_DATA, 4 * 4 * 512))
        test_features = np.reshape(test_features, (self.COUNT_TEST_DATA, 4 * 4 * 512))

        # перемешаем обучающую выборку для лучшей тренированости НС:
        indeces = np.random.choice(train_features.shape[0], size=train_features.shape[0],
                                   replace=False)
        train_features = train_features[indeces]

        # Теперь можно определить свой полносвязный классификатор (обратите внимание
        # на то, что для регуляризации здесь используется прием прореживания) и обучить
        # его на только что записанных данных и метках:
        self.model = keras.Sequential([
            Dense(256, activation='relu', input_dim=(4 * 4 * 512)),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

        # подготавливаем модель собственного компилятора к обучению:
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # обучаем классификатор
        self.history = self.model.fit(train_features, train_labels, epochs=50, batch_size=10,
                                      validation_data=(validation_features, validation_labels))

    def show_loss_acc_graphics(self):
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
        plt.show()

        '''БЫСТРОЕ ВЫДЕЛЕНИЕ ПРИЗНАКОВ БЕЗ РАСШИРЕНИЯ ДАННЫХ.
        Запись вывода conv_base в ответ на передачу наших данных и его использование
        в роли входных данных новой модели.'''

    def extract_features(self, directory, sample_count):
        '''Функция для выделения признаков из изображений товара'''

        # Создание матрицы признаков
        features_matrix = np.zeros(shape=(sample_count, 4, 4, 512))

        # Создание матрицы меток
        labels_matrix = np.zeros(shape=(sample_count))

        # создаем генератор для автоматического преобразования
        # файлов с изображениями в пакеты готовых тензоров
        generator = self.DATAGEN.flow_from_directory(
            directory,
            target_size=(150, 150),
            batch_size=self.BATCH_SIZE,
            class_mode='binary'
        )
        i = 0
        for inputs_batch, labels_batch in generator:
            # выделим признаки из изображений
            features_batch = self.conv_base.predict(inputs_batch)

            # помещаем выделенные признаки в матрицу
            features_matrix[i * self.BATCH_SIZE:(i + 1) * self.BATCH_SIZE] = features_batch

            # тоже самое делаем с метками
            labels_matrix[i * self.BATCH_SIZE: (i + 1) * self.BATCH_SIZE] = labels_batch
            i += 1
            if i * self.BATCH_SIZE >= sample_count:
                break
        return features_matrix, labels_matrix

    def get_predict(self, user_img):  #эта функция пока что не работает. Нужно исправление!
        '''Функция для предсказания объекта на пользовательском изображении'''
        # преобразовываем фото пользователя
        user_img = self.extract_features(user_img, 1)
        user_img = np.reshape(user_img, (1, 4 * 4 * 512))

        # получаем прогноз. если перменная argmax принимает значение 0 ( 0 - это первый нейрон,
        # отвечающий за пиво),то га изображении пиво,
        # если 1 , то водка
        result = self.model.predict(user_img)
        print(result, np.argmax(result), sep='\n')
        if np.argmax(result) == 0:
            print('На фото "Оболонь Премиум" 1,1 л ')
        else:
            print('На фото водка "Гетьман" 0,7 л ')


user = GroceryAppPhoto()
beer_image ='/home/andrey/grocery_data/predicts'
vodka_image='/home/andrey/grocery_data/predicts/vodka.jpg'
#user.get_predict(beer_image)

