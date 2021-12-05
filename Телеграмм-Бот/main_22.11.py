import random
import matplotlib.pyplot as plt
import openpyxl
from sympy.plotting import plot3d
import telebot
from telebot import types
from sympy import *
from itertools import combinations_with_replacement as cwr
from requests import get
import pandas as pd
import numpy as np
import os
import io
import PIL
import warnings
warnings.filterwarnings('ignore')

Memes = ['https://docs.google.com/uc?id=16I6-Q4zpgbvRY9m3s4OqF3JcjSudBD-m',
         'https://docs.google.com/uc?id=1dGAkJcjjYAWrLizmRSz4sjflxUa4h8LY',
         'https://docs.google.com/uc?id=1R2ynD71DVymDYKbv9vfE7mx3M14dRS4g',
         'https://docs.google.com/uc?id=19ZBP1-CUFf38DuQ7BwtMzmAyf2h8FAXh',
         'https://docs.google.com/uc?id=1xJYn5knfx-Lp5ncXViXfP1UPP1daSmbX',
         'https://docs.google.com/uc?id=1LCtMf1OiIB8kvjRjofUznBmuMhWWNcOB',
         'https://docs.google.com/uc?id=191Q3wQR_LVkYd9LUGfOCIiH4d1ofnaxh',
         'https://docs.google.com/uc?id=1iD6vXqLZxJY64L6e846fKlEC9YekztPR',
         'https://docs.google.com/uc?id=1Mz5776LM95GD8DRdFQAVcFszvQ_mO10t',
         'https://docs.google.com/uc?id=10BetU6QKDC5tOwkVikPd5_EIzl2opPwr',
         'https://docs.google.com/uc?id=17UBRefMwZRgXgipEIaXgUl0ZFYP29GLD',
         'https://docs.google.com/uc?id=1DKz-apMshVhYU_vGwCs0ip0XjJSK6PSz',
         'https://docs.google.com/uc?id=1tBvewivJwUzlyXJBlPPlqqbgDrz2MqfL',
         'https://docs.google.com/uc?id=1lr8tMjGzAOm8VZyjk5JzPjs8ilj3Wohz',
         'https://docs.google.com/uc?id=1uODPV7z_OYkuSyqpsbT0myINuW5d4P_B',
         'https://docs.google.com/uc?id=1iy2iClqzwzRkIIa5QuX1wF8ymmVb9qQW',
         'https://docs.google.com/uc?id=187A8u4zIl2iGJTV3bB-boSUSlRPUMQt_',
         'https://docs.google.com/uc?id=187A8u4zIl2iGJTV3bB-boSUSlRPUMQt_']

# подключим токен нашего бота
bot = telebot.TeleBot("***************************")
token = "***************************"

# Напишем функцию для старта:
@bot.message_handler(commands=['start'])
def send_keyboard(message, text="Привет, Добрый человек! Если ты впервые пользуешься моими услугами, "
                                "то настоятельно рекомендую прочитать раздел Информация, где подробно расписаны "
                                "правила ввода и многое другое! Если же ты уже все знаешь, выбирай раздел:"):
    # bot.send_message()
    keyboard = types.ReplyKeyboardMarkup(row_width=1)
    it_1 = types.KeyboardButton("Матан")
    it_2 = types.KeyboardButton("Линал")
    it_4 = types.KeyboardButton("Построение графиков")
    it_5 = types.KeyboardButton("Справочные материалы")
    it_6 = types.KeyboardButton("Информация")
    keyboard.add(it_1, it_2, it_4, it_5, it_6)
    answer = bot.send_message(message.chat.id, text=text, reply_markup=keyboard)
    bot.register_next_step_handler(answer, choose_section)


@bot.message_handler(content_types=['text'])
def handle_docs_audio(message):
    send_keyboard(message, text="Я не понимаю :-( Давайте начнем с начала! \n Выберите раздел математики:")


# Функция выбора подраздела математики:
def choose_section(answer):
    if answer.text == "Матан":
        # Создаем клавиатуру
        keyboard = types.ReplyKeyboardMarkup()
        it_1 = types.KeyboardButton("Производные")
        it_2 = types.KeyboardButton("Интеграл")
        it_3 = types.KeyboardButton("Диффуры")
        it_5 = types.KeyboardButton("Главное меню")
        keyboard.add(it_1, it_2, it_3, it_5)
        # Присылаем ее пользователю с просбой выбора
        answer = bot.send_message(answer.chat.id, "Выберите раздел Матана", reply_markup=keyboard)
        bot.register_next_step_handler(answer, func_matan)


    elif answer.text == "Линал":

        # Создаем клавиатуру
        keyboard = types.ReplyKeyboardMarkup(row_width=4)
        it_1 = types.KeyboardButton("min, max, rank")
        it_2 = types.KeyboardButton("det(A)")
        it_3 = types.KeyboardButton("A.T, A^-1")
        it_4 = types.KeyboardButton("A+B")
        it_5 = types.KeyboardButton("A-B")
        it_6 = types.KeyboardButton("A*B")
        it_7 = types.KeyboardButton("Главное меню")
        keyboard.add(it_1, it_2, it_3, it_4, it_5, it_6, it_7)
        # Краткая справка по работе
        bot.send_message(answer.chat.id, 'Для операций с матрицами пришлите в формате xlsx в таком виде:')
        bot.send_photo(answer.chat.id, get("https://docs.google.com/uc?id=1D7BMF7TjVyNM-NhSp0tiF4sLbd9F5zuT").content)
        bot.send_message(answer.chat.id,
                         'Для операций с двумя матрицами пришлите файл excel c матрицами на разных листах; \n'
                         'Для опреаций с одной матрицей можно ввести её с клавиатуры.\n'
                         'Чтобы ввести матрицу в бот, ввведите элементы через пробел построчно, например: \n'
                         '1 2\n3 4')
        # Присылаем клавиатуру пользователю с просьбой выбора
        answer = bot.send_message(answer.chat.id, "Выберите раздел Линала", reply_markup=keyboard)
        bot.register_next_step_handler(answer, func_linal)
    #     func_linal - обрабатыввает функции линала!

    elif answer.text == "Построение графиков":
        # Создаем клавиатуру
        keyboard = types.ReplyKeyboardMarkup(row_width=2)
        it_2 = types.KeyboardButton("По функции")
        it_3 = types.KeyboardButton("Главное меню")
        keyboard.add(it_2, it_3)
        # Присылаем ее пользователю с просбой выбора
        answer = bot.send_message(answer.chat.id, "Выберите способ построения", reply_markup=keyboard)
        bot.register_next_step_handler(answer, func_graph)
    #     func_graph - обрабатыввает функции линала! Яна - это твой раздел)

    elif answer.text == "Справочные материалы":
        # Создаем клавиатуру
        keyboard = types.ReplyKeyboardMarkup(row_width=2)
        it_1 = types.KeyboardButton("Таблица производных")
        it_2 = types.KeyboardButton("Интегралы")
        it_3 = types.KeyboardButton("Свойства логарифмов")
        it_4 = types.KeyboardButton("Тригонометрия")
        it_5 = types.KeyboardButton("Свойства еще чего-нибудь")
        it_6 = types.KeyboardButton("Мемы)))")
        it_7 = types.KeyboardButton("Главное меню")
        keyboard.add(it_1, it_2, it_3, it_4, it_5, it_6, it_7)
        # Присылаем ее пользователю с просбой выбора
        answer = bot.send_message(answer.chat.id, "Выберите раздел", reply_markup=keyboard)
        bot.register_next_step_handler(answer, func_spravka)
    #     func_spravka - отсылает все картинки по запросам))
    #     Пока все картинки подгружаю со своего компа, но я их загружу на гугл диск

    elif answer.text == "Информация":
        bot.send_message(answer.chat.id, "Здесь надо прислать всю инфу о канале")
        send_keyboard(answer, 'Выберите раздел')
    else:
        send_keyboard(answer, 'Пожалуйста, выбери раздел:')


# Пишем функции для обработки разделов!


def func_matan(message):
    if message.text == 'Производные':
        print(message)
        f = bot.send_message(message.chat.id, 'Введите функцию, переменные и степень производной, начиная каждый раз '
                                              'с новой строки \nНапример: '
                                              '\nsin(x)*cos(y) \nx y \n5 \n '
                                              '\n P.S. Пришлите все одним сообщением, пожалуйста!')

        bot.register_next_step_handler(f, derivative)
    elif message.text == 'Интеграл':
        f = bot.send_message(message.chat.id, 'Введите функцию, переменные интегрирования и пределы интегрирования'
                                              '(через запятую), начиная каждый раз с новой строки\n \nНапример: '
                                              '\nsin(x)*cos(y) \nx, 0, pi/2\ny, 0, pi/2 \n \n P.S. Пришлите все одним '
                                              'сообщением, пожалуйста! \n P.S.S. Если вам нужен '
                                              'неопределенный интеграл, напишите -1 в пределах')
        bot.register_next_step_handler(f, diff_eq)
    elif message.text == 'Диффуры':
        f = bot.send_message(message.chat.id, 'Блаблабла')
        bot.register_next_step_handler(f, integral)

    elif message.text == 'Главное меню':
        send_keyboard(message, 'Порешаем еще?')
    else:
        msg = bot.send_message(message.chat.id, 'Я не понимаю(((')
        send_keyboard(msg, 'Выберите раздел')

def derivative(info):
    calculations = derivative_calc(info.text)
    if calculations == 0:
        send_keyboard(info, 'Данные введены неверно, я не понимаю((((')
    else:
        # Рисуем картинки и отправляем
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        t = ax.text(0.5, 0.5, '\n'.join(calculations),
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=20, color='black')
        ax.figure.canvas.draw()
        bbox = t.get_window_extent()
        fig.set_size_inches(bbox.width / 80, bbox.height / 80)  # dpi=80

        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = PIL.Image.open(buf)
        try:
            # Отправляем фото пользователю
            bot.send_photo(info.chat.id, img)

        except:
            bot.send_message(info.chat.id, 'К сожалению, мы не смогли отправить Вам ответ, '
                                           'так как у нас возникли неполадки, но мы уже работаем над этим!')
            bot.send_message(info.chat.id, 'Чтобы Вы не огорчались, мы отправим вам мем =) :')
            i = random.randint(0, 17)
            bot.send_photo(info.chat.id, get(f'{Memes[i]}').content)
        send_keyboard(info, 'Порешаем еще?')


def derivative_calc(data):
    try:
        function = data.split('\n')[0].lower()  # ЗАМЕНИ _ НА \n
        variable = (data.split('\n')[1].lower()).split()  # ЗАМЕНИ _ НА \n
        degree = int(data.split('\n')[2])  # ЗАМЕНИ _ НА \n
        deriv = list(cwr(variable, degree))
        degree_latex = [(''.join(list(el))) for el in deriv]
        answer = [f"diff({function}, {', '.join(el)})" for el in deriv]

        # Sympify нам все посчитает и так же вернет списко
        answer = [sympify(el) for el in answer]
        # Превращаем ответ в текст латеха
        answer_latex = [f"$f_{degree_latex[i]} = {latex(answer[i])}$" for i in range(len(answer))]
        # Для красоты добавляем вверху запись введенной функции
        answer_latex.insert(0, f'$f({", ".join(variable)}) = {function}$')
    except:
        answer_latex = 0
    return answer_latex


def integral(info):
    calculations = integral_calc(info.text)
    if calculations == 0:
        send_keyboard(info, 'Данные введены неверно, я не понимаю((((')
    else:
        # Рисуем картинки и отправляем
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        t = ax.text(0.5, 0.5, calculations,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=20, color='black')
        ax.figure.canvas.draw()
        bbox = t.get_window_extent()
        fig.set_size_inches(bbox.width / 80, bbox.height / 80)  # dpi=80

        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = PIL.Image.open(buf)
        try:
            # Отправляем фото пользователю
            bot.send_photo(info.chat.id, img)
        except:
            bot.send_message(info.chat.id, 'К сожалению, мы не смогли отправить Вам ответ, '
                                           'так как у нас возникли неполадки, но мы уже работаем над этим!')
            bot.send_message(info.chat.id, 'Чтобы Вы не огорчались, мы отправим вам мем =) :')
            i = random.randint(0, 17)
            bot.send_photo(info.chat.id, get(f'{Memes[i]}').content)
        send_keyboard(info, 'Порешаем еще?')


def integral_calc(data):
    try:
        list = data.split('\n')
        function = sympify(list[0].lower())
        variable = []
        if len(list) == 2:
            variable = [list[1].lower()]
        elif len(list) == 3:
            variable = [list[1].lower(), list[2].lower()]
        v_withot_lim = [el[0] for el in variable]
        # Формируем строку для расчета simpy и сразу считаем через sympify
        answer = sympify(f"integrate({function}, ({'), ('.join(variable)}))")
        # А здесь мы формируем красивую печать интеграла)))
        expression = sympify(f"Integral({function}, ({'), ('.join(v_withot_lim)}))")
        # Формируем ответ в виде латеха
        answer_latex = fr"${latex(expression)} = {latex(answer)} + C$"
    except:
        answer_latex = 0
    return answer_latex

from function import diffEq_calc
def diff_eq(info):
    calculations = diffEq_calc(info.text)
    if calculations == 0:
        send_keyboard(info, 'Данные введены неверно, я не понимаю((((')
    else:
        # Рисуем картинки и отправляем
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        t = ax.text(0.5, 0.5, calculations,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=20, color='black')
        ax.figure.canvas.draw()
        bbox = t.get_window_extent()
        fig.set_size_inches(bbox.width / 80, bbox.height / 80)  # dpi=80

        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = PIL.Image.open(buf)
        try:
            # Отправляем фото пользователю
            bot.send_photo(info.chat.id, img)
        except:
            bot.send_message(info.chat.id, 'К сожалению, мы не смогли отправить Вам ответ, '
                                           'так как у нас возникли неполадки, но мы уже работаем над этим!')
            bot.send_message(info.chat.id, 'Чтобы Вы не огорчались, мы отправим вам мем =) :')
            i = random.randint(0, 17)
            bot.send_photo(info.chat.id, get(f'{Memes[i]}').content)
        send_keyboard(info, 'Порешаем еще?')


# ФУНКЦИИ ЛИНАЛА
def func_linal(message):
    if message.text == 'min, max, rank':
        message = bot.send_message(message.chat.id, 'Пришлите документ .xlsx или напишите матрицу здесь:')
        bot.register_next_step_handler(message, mmr)
    elif message.text == 'det(A)':
        bot.send_message(message.chat.id, 'Пришлите документ .xlsx или напишите матрицу здесь:')
        bot.register_next_step_handler(message, detm)
    elif message.text == 'A.T, A^-1':
        bot.send_message(message.chat.id, 'Пришлите документ .xlsx:')
        bot.register_next_step_handler(message, tinv)
    elif message.text == 'A+B':
        bot.send_message(message.chat.id, 'Пришлите документ .xlsx:')
        bot.register_next_step_handler(message, msum)
    elif message.text == 'A-B':
        bot.send_message(message.chat.id, 'Пришлите документ .xlsx:')
        bot.register_next_step_handler(message, mdif)
    elif message.text == 'A*B':
        bot.send_message(message.chat.id, 'Пришлите документ .xlsx:')
        bot.register_next_step_handler(message, mmul)
    elif message.text == 'Главное меню':
        send_keyboard(message, 'Порешаем еще?')
    else:
        msg = bot.send_message(message.chat.id, 'Я не понимаю(((')
        send_keyboard(msg, 'Выберите раздел')


# Принимает матрицу из файла, находит max, min элементы, ранг
# Обработчик для документов
@bot.message_handler(content_types=(['document'], ['text']))
def mmr(message):
    # проверяем, что нам прислали, работаем с документом
    if message.text is None:
        # сохраняем file_id (идентификатор) присланного файла
        document_id = message.document.file_id
        # получаем путь до файла
        file_info = bot.get_file(document_id)
        # сохраняем матрицу из excel в pandas dataframe с помощью api из сохранённого пути
        # сразу берём только значения матрицы, которые будут в виде np.array
        try:
            mat = pd.read_excel(f'http://api.telegram.org/file/bot{token}/{file_info.file_path}', header=None,
                                sheet_name=0).dropna(how='all').dropna(axis=1).values
            # отправляем сообщения пользователю с информацией
            bot.send_message(message.chat.id, f'Максимальный элемент: {mat.max()}')
            bot.send_message(message.chat.id, f'Минимальный элемент: {mat.min()}')
            bot.send_message(message.chat.id, f'rank(A) = {np.linalg.matrix_rank(mat)}')
        except:
            bot.send_message(message.chat.id, 'К сожалению, мы не смогли отправить Вам ответ, '
                                              'так как у нас возникли неполадки, но мы уже работаем над этим!')
            bot.send_message(message.chat.id, 'Чтобы Вы не огорчались, мы отправим вам мем =) :')
            i = random.randint(0, 17)
            bot.send_photo(message.chat.id, get(f'{Memes[i]}').content)
    else:
        # на случай, если начали вводить сообщение
        # bot.send_message(message.chat.id, f'Мы работаем над этой функцией\n Ваша матрица: \n {message.text}')
        # преобразуем строки в матрицу
        m = np.array([i.split() for i in message.text.split('\n')]).astype(int)
        bot.send_message(message.chat.id, f'Максимальный элемент: {m.max()}')
        bot.send_message(message.chat.id, f'Минимальный элемент: {m.min()}')
        bot.send_message(message.chat.id, f'rank(A) = {np.linalg.matrix_rank(m)}')
    # после предлагаем выбрать другие функции
    send_keyboard(message, 'Выбирай раздел')


# Принимает матрицу из файла, считает определитель
@bot.message_handler(content_types=(['document'], ['text']))
def detm(message):
    if message.text is None:
        document_id = message.document.file_id
        file_info = bot.get_file(document_id)
        try:
            mat = pd.read_excel(f'http://api.telegram.org/file/bot{token}/{file_info.file_path}', header=None,
                                sheet_name=0).dropna(how='all').dropna(axis=1).values
            try:
                detm = np.linalg.det(mat)
                bot.send_message(message.chat.id, 'det(A) = {:.1f}'.format(detm))
            except:
                bot.send_message(message.chat.id, 'Чтобы посчитать определитель, введи квадратную матрицу')
                bot.send_message(message.chat.id, f'Размерность данной матрциы: {mat.shape}')
        except:
            bot.send_message(message.chat.id, 'К сожалению, мы не смогли отправить Вам ответ, '
                                              'так как у нас возникли неполадки, но мы уже работаем над этим!')
            bot.send_message(message.chat.id, 'Чтобы Вы не огорчались, мы отправим вам мем =) :')
            i = random.randint(0, 17)
            bot.send_photo(message.chat.id, get(f'{Memes[i]}').content)
    else:
        # bot.send_message(message.chat.id, f'Мы работаем над этой функцией\n Ваша матрица: \n {message.text}')
        m = np.array([i.split() for i in message.text.split('\n')]).astype(int)
        try:
            detm = np.linalg.det(m)
            bot.send_message(message.chat.id, 'det(A) = {:.1f}'.format(detm))
        except:
            bot.send_message(message.chat.id, 'Чтобы посчитать определитель, введи квадратную матрицу')
            bot.send_message(message.chat.id, f'Размерность данной матрциы: {m.shape}')
    send_keyboard(message, 'Выбирай раздел')


# Принимает матрицу из файла, транспонирует и считает обратную и выводит их на разных листах нового файла или в чат
@bot.message_handler(content_types=(['document'], ['text']))
def tinv(message):
    if message.text is None:
        document_id = message.document.file_id
        file_info = bot.get_file(document_id)
        try:
            mat = pd.read_excel(f'http://api.telegram.org/file/bot{token}/{file_info.file_path}', header=None,
                                sheet_name=0).dropna(how='all').dropna(axis=1).values
            t = np.round(mat.T)

            # в зависимости от размера матрицы отправляем сообщением или файлом
            if mat.shape[0] <= 10 or mat.shape[1] <= 10:
                # преобразуем матрицу в строки с элементами через пробел и выведем построчно
                n = [list(t[i]) for i in range(t.shape[0])]
                nn = "\n".join([" ".join(str(i).strip('[]').split(',')) for i in n])
                bot.send_message(message.chat.id, f'Транспонированная матрица: \n{nn}')
            else:
                df_t = pd.DataFrame(t)
                # file = df_t.to_excel('calc_excel.xlsx', sheet_name='Transp matr', header=False, index=False)

            if mat.shape[0] <= 10 or mat.shape[1] <= 10:
                try:
                    inv = np.round(np.linalg.inv(mat), 1)
                    n = [list(inv[i]) for i in range(inv.shape[0])]
                    nn = "\n".join([" ".join(str(i).strip('[]').split(',')) for i in n])
                    bot.send_message(message.chat.id, f'Обратная матрица: \n{nn}')
                except:
                    bot.send_message(message.chat.id, 'Матрица вырождена -> обратной нет')
            else:
                df_t = pd.DataFrame(t)
                # try:
                inv = np.round(np.linalg.inv(mat), 1)
                df_inv = pd.DataFrame(inv)
                # file = df_t.to_excel('calc_excel.xlsx', sheet_name='Inv matr', header=False, index=False)
                buf = io.BytesIO()
                buf.name = "answer.xlsx"
                with pd.ExcelWriter(buf) as writer:
                    df_t.to_excel(writer, sheet_name='Transp matr', header=False, index=False)
                    df_inv.to_excel(writer, sheet_name='Inv matr', header=False, index=False)
                buf.seek(0)
                bot.send_document(message.chat.id, buf)
                # docs = openpyxl.open(buf)
                # print(type(docs))


                    # with pd.ExcelWriter('calc_excel.xlsx') as writer:
                    #     df_t.to_excel(writer, sheet_name='Transp matr', header=False, index=False)
                    #     df_inv.to_excel(writer, sheet_name='Inv matr', header=False, index=False)
                # bot.send_document(message.chat.id, docs)
                # except:
                #     with pd.ExcelWriter('calc_excel.xlsx') as writer:
                #         df_t.to_excel(writer, sheet_name='Transp matr', header=False, index=False)
                #     bot.send_document(message.chat.id, open("calc_excel.xlsx", 'rb'))
                #     bot.send_message(message.chat.id, 'Матрица вырождена -> обратной нет')
        except IndexError:
            bot.send_message(message.chat.id, 'К сожалению, мы не смогли отправить Вам ответ, '
                                              'так как у нас возникли неполадки, но мы уже работаем над этим!')
            bot.send_message(message.chat.id, 'Чтобы Вы не огорчались, мы отправим вам мем =) :')
            i = random.randint(0, 17)
            bot.send_photo(message.chat.id, get(f'{Memes[i]}').content)
    else:
        bot.send_message(message.chat.id, f'Мы работаем над этой функцией\n Ваша матрица: \n {message.text}')
    send_keyboard(message, 'Выбирай раздел')


# Принимает 2 матрицы с разных листов excel и складывает их
@bot.message_handler(content_types=(['document'], ['text']))
def msum(message):
    if message.text is None:
        document_id = message.document.file_id
        file_info = bot.get_file(document_id)
        try:
            mat1 = pd.read_excel(f'http://api.telegram.org/file/bot{token}/{file_info.file_path}', header=None,
                                 sheet_name=0).dropna(how='all').dropna(axis=1).values
            mat2 = pd.read_excel(f'http://api.telegram.org/file/bot{token}/{file_info.file_path}', header=None,
                                 sheet_name=1).dropna(how='all').dropna(axis=1).values
            if mat1.shape == mat2.shape:
                m = mat1 + mat2
                n = [list(m[i]) for i in range(m.shape[0])]
                nn = "\n".join([" ".join(str(i).strip('[]').split(',')) for i in n])
                bot.send_message(message.chat.id, f'A + B: \n{nn}')
            else:
                bot.send_message(message.chat.id, 'Размерности матриц не совпадают')
        except:
            bot.send_message(message.chat.id, 'К сожалению, мы не смогли отправить Вам ответ, '
                                              'так как у нас возникли неполадки, но мы уже работаем над этим!')
            bot.send_message(message.chat.id, 'Чтобы Вы не огорчались, мы отправим вам мем =) :')
            i = random.randint(0, 17)
            bot.send_photo(message.chat.id, get(f'{Memes[i]}').content)
    else:
        bot.send_message(message.chat.id, f'Мы работаем над этой функцией\n Ваша матрица: \n {message.text}')
    send_keyboard(message, 'Выбирай раздел')


# Принимает 2 матрицы с разных листов excel и находит их разность
@bot.message_handler(content_types=(['document'], ['text']))
def mdif(message):
    if message.text is None:
        document_id = message.document.file_id
        file_info = bot.get_file(document_id)
        try:
            # сохраняем 2 разные матрицы с разных листов excel
            mat1 = pd.read_excel(f'http://api.telegram.org/file/bot{token}/{file_info.file_path}', header=None,
                                 sheet_name=0).dropna(how='all').dropna(axis=1).values
            mat2 = pd.read_excel(f'http://api.telegram.org/file/bot{token}/{file_info.file_path}', header=None,
                                 sheet_name=1).dropna(how='all').dropna(axis=1).values
            if mat1.shape == mat2.shape:
                m = mat1 - mat2
                n = [list(m[i]) for i in range(m.shape[0])]
                nn = "\n".join([" ".join(str(i).strip('[]').split(',')) for i in n])
                bot.send_message(message.chat.id, f'A - B: \n{nn}')
            else:
                bot.send_message(message.chat.id, 'Размерности матриц не совпадают')
        except:
            bot.send_message(message.chat.id, 'К сожалению, мы не смогли отправить Вам ответ, '
                                              'так как у нас возникли неполадки, но мы уже работаем над этим!')
            bot.send_message(message.chat.id, 'Чтобы Вы не огорчались, мы отправим вам мем =) :')
            i = random.randint(0, 17)
            bot.send_photo(message.chat.id, get(f'{Memes[i]}').content)
    else:
        bot.send_message(message.chat.id, f'Мы работаем над этой функцией\n Ваша матрица: \n {message.text}')
    send_keyboard(message, 'Выбирай раздел')


# Принимает 2 матрицы с разных листов excel и перемножает их
@bot.message_handler(content_types=(['document'], ['text']))
def mmul(message):
    if message.text is None:
        document_id = message.document.file_id
        file_info = bot.get_file(document_id)
        try:
            mat1 = pd.read_excel(f'http://api.telegram.org/file/bot{token}/{file_info.file_path}', header=None,
                                 sheet_name=0).dropna(how='all').dropna(axis=1).values
            mat2 = pd.read_excel(f'http://api.telegram.org/file/bot{token}/{file_info.file_path}', header=None,
                                 sheet_name=1).dropna(how='all').dropna(axis=1).values
            try:
                m = mat1 @ mat2
                n = [list(m[i]) for i in range(m.shape[0])]
                nn = "\n".join([" ".join(str(i).strip('[]').split(',')) for i in n])
                bot.send_message(message.chat.id, f'A*B: \n{nn}')
            except:
                bot.send_message(message.chat.id, 'Размерности матриц не совпадают')
        except:
            bot.send_message(message.chat.id, 'К сожалению, мы не смогли отправить Вам ответ, '
                                              'так как у нас возникли неполадки, но мы уже работаем над этим!')
            bot.send_message(message.chat.id, 'Чтобы Вы не огорчались, мы отправим вам мем =) :')
            i = random.randint(0, 17)
            bot.send_photo(message.chat.id, get(f'{Memes[i]}').content)
    else:
        bot.send_message(message.chat.id, f'Мы работаем над этой функцией\n Ваша матрица: \n {message.text}')
    send_keyboard(message, 'Выбирай раздел')

# отрисовка графиков
def func_graph(message):
    if message.text == 'По функции':
        f = bot.send_message(message.chat.id, 'Введите вашу функцию и переменные через пробел:\n'
                                              'Например: cos(x)*sin(y) x y ')
        bot.register_next_step_handler(f, graph)
    elif message.text == 'Главное меню':
        send_keyboard(message, 'Порешаем еще?')
    else:
        msg = bot.send_message(message.chat.id, 'Я не понимаю(((')
        send_keyboard(msg, 'Выберите раздел')


def graph(info):
    # Делаем фотку и сохраняем ее
    c = plot_graph(info.text, info.chat.id)
    if c == 0:
        send_keyboard(info, 'Данные введены неверно')
    elif c == 1:
        try:
            bot.send_photo(info.chat.id, photo=open(f'{info.chat.id}.png', 'rb'))
            # Удаляемм фото
            path = os.path.join(os.path.abspath(os.path.dirname(__file__)), f"{info.chat.id}.png")
            os.remove(path)
        except:
            bot.send_message(info.chat.id, 'Извините, у нас неполадки, мы работаем над этим!')
        send_keyboard(info, 'Главное меню')


def plot_graph(data, id):
    try:
        list_1 = data.lower().split()
        print(list_1)
        if len(list_1) < 2:
            ret = 0  # Данные введены неверно
        else:
            function = sympify(list_1.pop(0))
            if len(list_1) == 1:
                plt_1 = plot(function, show=False)
                # plt_1.show()
                plt_1.save(f'{id}.png')
                ret = 1
            elif len(list_1) == 2:
                plt_2 = plot3d(function, show=False)
                plt_2.save(f'{id}.png')
                ret = 1
            else:
                ret = 0
    except:
        ret = 0
    return ret


def func_spravka(message):
    if message.text == 'Таблица производных':
        bot.send_photo(message.chat.id, 'https://docs.google.com/uc?id=1tRpWTHF2ZK06r3YY5BwUEOi6p0Mwx5l9')
        send_keyboard(message, 'Выбирай раздел')
    elif message.text == 'Интегралы':
        bot.send_photo(message.chat.id, 'https://docs.google.com/uc?id=1x-9XZEwpIkKdUtcWKGt3fkdL39koDd8s')
        send_keyboard(message, 'Выбирай раздел')
    elif message.text == 'Свойства логарифмов':
        bot.send_photo(message.chat.id, 'https://docs.google.com/uc?id=1nQBFrt4h3yF8mrqLN0-mQ9Mdi8MNMqxR')
        send_keyboard(message, 'Выбирай раздел')
    elif message.text == 'Тригонометрия':
        bot.send_photo(message.chat.id, 'https://docs.google.com/uc?id=1VN_2pBQfiPXO1J0MHiFlb9RqXhEymwpC')
        send_keyboard(message, 'Выбирай раздел')
    elif message.text == 'Свойства еще чего-нибудь':
        bot.send_photo(message.chat.id, 'https://docs.google.com/uc?id=1Kw9t8be9q0vootydiHhg2Ed75Dbq4YpD')
        send_keyboard(message, 'Выбирай раздел')
    elif message.text == 'Мемы)))':
        i = random.randint(0, 17)
        bot.send_photo(message.chat.id, get(f'{Memes[i]}').content)
        send_keyboard(message, 'Выбирай раздел')
    elif message.text == 'Главное меню':
        send_keyboard(message, ' Может теперь сам порешаешь?')
    else:
        msg = bot.send_message(message.chat.id, 'Я не понимаю(((')
        send_keyboard(msg, 'Выберите раздел')


def photo(tex, id):
    # Здесь в переменную tex мы подаем строку с текстом латеха!!!
    # Создаем область Рисунка
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    # ax.set_axis_off()
    #  Рисуем формулы
    t = ax.text(0.5, 0.5, tex,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=20, color='black')
    # Определение размеров формулы
    ax.figure.canvas.draw()
    bbox = t.get_window_extent()

    # Установка размеров рисунка
    fig.set_size_inches(bbox.width / 80, bbox.height / 80)  # dpi=80

    # Сохранение формулы в файл с номером id чата
    plt.savefig(f"photo_{id}.png", dpi=300)
    # plt.show()

bot.polling()
# bot.polling(none_stop=True)