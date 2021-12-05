import sympy
from sympy import *
from itertools import combinations_with_replacement as cwr
from itertools import groupby as gr
import matplotlib.pyplot as plt
from sympy.plotting import plot3d
# Подклюяаем библиотеки Латекса, чтобы матплот мог их юзать))
# plt.rc('text', usetex=True)



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

def derivative_calc(data):
    try:
        function = data.split('\n')[0].lower() #ЗАМЕНИ _ НА \n
        variable = (data.split('\n')[1].lower()).split() #ЗАМЕНИ _ НА \n
        degree = int(data.split('\n')[2]) #ЗАМЕНИ _ НА \n
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

def diffEq_calc(data):
    try:
        # Добавить строку, что мы умеем решеать,а шо нет)))
        # ВВод след вида y'' + 2y' - 3 = sin(x) \n y max 3 порядок!!!!
        data = data.lower()
        data = data.replace("\"", "''")  # превращаем " в '
        # Записываем дифференцируемую функцию в переменную f
        function = data.split('\n')[1].replace(" ", "")
        # Убираем все повторы, чтобы не было ошибок при косяках на вводе
        function = ''.join(ch for ch, _ in gr(function))
        f = symbols(function[0], cls=Function)
        # Записываем в х независимую переменную, находящуюся в скобках
        x = symbols(function[4: len(function) - 1])
        # В переменной data хранится уравнение
        data = data.split('\n')[0]
        data = ''.join(data.split())  # убираем все пробелы
        # Ищем какие производные есть в уравнении
        hatch_list = []
        copy = data
        if copy.find("'" * 4) != -1:
            answer_latex = '1'  # "Я умею решать уравнения максимум 3 порядка(((")
        else:
            for i in range(3, 0, -1):
                if copy.find("'" * i) != -1:
                    hatch_list.append(i)
                    copy = copy.replace("'" * i, "")

            # Преобразуем все это дело в виде функции sympy
            for el in hatch_list:
                data = data.replace("\'" * el, f'({x}).diff({x}, {el})')
            data = data.replace('=', ',')
            equation = sympify(f'Eq({data})')
            # А тепееерь, магия sympy)))
            answer_latex = f"${latex(dsolve(equation, f(x)))}$"
    except:
        answer_latex = '0'
    return answer_latex


def plot_graph(data, id):
    try:
        list_1 = data.lower().split()
        print(list_1)
        if len(list_1) < 2:
            ret = 0 # Данные введены неверно
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











# Просто дала име эксельке и отправила в виде двоичной формы. проверить у Алены

# file_obj = io.BytesIO(your_bin_data)
# file_obj.name = "MarksSYAP.xlsx"
# bot.send_document(message.chat.id,file_obj)
