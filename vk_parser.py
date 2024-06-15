import requests
import json
import os
import hashlib
import hashlib
from keras.utils import image_dataset_from_directory
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from keras.applications import ConvNeXtLarge
from keras.applications import MobileNet
from keras.applications import EfficientNetV2L
from keras.applications import  EfficientNetB7
import tensorflow as tf
from PIL import Image

#https://vk.com/public139096133


# Здесь должен содержаться токен для возможности запросов Вк
# Узнать его можно с помощью туториала https://www.youtube.com/watch?v=qxPMBBULFbs
# Только нужно создавать не stand-alone приложение, а web-приложение
TOKEN = #//////////
ID = -139096133 # ID паблика, который будем парсить, пишется через -
SAVE_PATH = './SOUL/'
COUNT = 100 # Число запросов
PATH_OF_ROW_DATASET = './SOUL/'
BATCH_SIZE = 32
i = 0

#                                                {массив постов}    {массив фотографий}       {последний из размеров}
#print(json.loads(response.content)['response']['items'][6]['attachments'][-1]['photo']['sizes'][-1]['url'])

def save_picture(url):
    """Загрузка одного изображения по его ссылке"""
    global i
    response = requests.get(url)
    if response.status_code == 200:
        with open(SAVE_PATH + f'{i}.jpg', 'wb') as pic:
            pic.write(response.content)
        i += 1


def get_count_posts():
    """Возвращает количество постов в паблике по ID"""
    URL = f'https://api.vk.com/method/wall.get?owner_id={ID}&count=1&access_token={TOKEN}&v=5.131'
    response = requests.get(URL)
    data = response.json()
    if 'response' in data and 'count' in data['response']:
        post_count = data['response']['count']
        return post_count
    else:
        return None


def save_all_pictures():
    """Отправляем запросы и сохраняем все изображения по ссылкам из постов (до числа округленного до сотен) паблика по ID"""
    count_posts = get_count_posts()
    num_requests = count_posts // COUNT
    for offset in range(num_requests):
        URL = f'https://api.vk.com/method/wall.get?owner_id={ID}&access_token={TOKEN}&count={COUNT}&offset={COUNT * offset}&v=5.131'
        response = requests.get(URL)
        data = json.loads(response.content)
        if 'response' in data:
            for post in data['response']['items']:
                for photo in post['attachments']:
                    print(photo)
                    try:
                        # Достаем ссылку на последнюю по размеру картинку
                        # (Да-да, в вк несколько картинок по размеру скрыто в одном посте)
                        url_pic = photo['photo']['sizes'][-1]['url']
                        save_picture(url_pic)
                    except:
                        # Пропускаем если нет изображений в посте
                        continue


def delete_small_images(folder=PATH_OF_ROW_DATASET, min_width=256, min_height=256):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            width, height = (0, 0)
            with Image.open(file_path) as img:
                width, height = img.size
            if width < min_width or height < min_height:
                os.remove(file_path)


def delete_duplicates(duplicates, folder_from=PATH_OF_ROW_DATASET):
    """Удаляет изображения, названия которых находятся в duplicates"""
    duplicates = set(duplicates)
    dir = set(os.listdir(folder_from))
    for file in dir:
        if file in duplicates:
            os.remove(folder_from + file)


def find_duplicates(folder=PATH_OF_ROW_DATASET):
    """Возвращает лист названий дублирующих изображений в папке"""
    images = set()
    duplicate = set()
    for file in os.listdir(folder):
        image = Image.open(folder + file)
        hash_image = hashlib.sha1(image.tobytes()).hexdigest()
        if hash_image in images:
            duplicate.add(file)
        else:
            images.add(hash_image)

    return list(duplicate)


def classification_of_row_dataset(row_dataset=PATH_OF_ROW_DATASET, result_path='./dataset/', image_size=(256, 256)):
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    row_dataset = image_dataset_from_directory(
        row_dataset,
        labels=None,
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        image_size=image_size,
        shuffle=False,
    )

    file_names = row_dataset.file_paths

    row_dataset = row_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    model = EfficientNetV2L(
        weights='imagenet',
        include_top=True,
        input_shape=(480, 480, 3)
    )

    prepare_image = tf.keras.Sequential([
        tf.keras.layers.Resizing(480, 480),
    ])

    # ConvNeXtLarge - нормально справляется 2.5/4
    # EfficientNetV2L - хорошо 3.5/4
    # vgg19 - плохо
    for i, batch in enumerate(row_dataset):
        batch = prepare_image(batch)
        batch_classes = model.predict(batch)
        predictions = decode_predictions(batch_classes, top=1)

        for j, (picture, predict) in enumerate(zip(batch, predictions)):
            class_name = predict[0][1].lower()
            num_picture = i * BATCH_SIZE + j
            picture_name = file_names[num_picture].rsplit('/', 1)
            if class_name not in os.listdir(result_path):
                os.makedirs(f'{result_path}{class_name}/')
            image = tf.cast(batch[j], tf.uint8)
            filename = f'{result_path}{class_name}/{picture_name[-1]}'
            tf.io.write_file(filename, tf.image.encode_jpeg(image))

        tf.keras.backend.clear_session()



if __name__ == "__main__":
    save_all_pictures()
    delete_small_images(PATH_OF_ROW_DATASET)
    duplicates = find_duplicates('./SOUL/')
    delete_duplicates(duplicates, './SOUL/')
    classification_of_row_dataset('./SOUL/', image_size=(480, 480))
