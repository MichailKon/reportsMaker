import pandas as pd
import seaborn as sns
import base64
import numpy as np


def csv2html(df: pd.DataFrame, get_color=lambda x: None, drop_noise=True, name='', additional_info=None,
             draw_full=False, debug=False, convert_num=lambda x: round(float(x), 3), square_table=True, transpose=False,
             draw_columns=True, draw_rows=True, mark_biggest=False):
    if isinstance(convert_num, list) or isinstance(convert_num, tuple):
        pass
    else:
        convert_num = [convert_num for _ in range(df.shape[1])]

    if isinstance(get_color, list) or isinstance(get_color, tuple):
        pass
    else:
        get_color = [get_color for _ in range(df.shape[1])]

    sq_class = "class=\"square_table\"" if square_table else ""
    res = f"""<table {sq_class}>"""
    if name:
        res += f"<caption>{name}</caption>"

    if drop_noise:
        try:
            df = df.drop(-1)
        except KeyError:
            pass
        try:
            df = df.drop("-1")
        except KeyError:
            pass
        try:
            df = df.drop(-1, axis=1)
        except KeyError:
            pass
        try:
            df = df.drop('-1', axis=1)
        except KeyError:
            pass
    # get header
    if transpose:
        df = df.transpose()

    if draw_columns:
        if draw_rows:
            res += "<tr><th></th>"
        for i in df.columns:
            res += f"<th><div class=\"content\">{str(i)}</div></th>"
        res += "</tr>\n"
    #
    try:
        max_val = np.array(df).max()
    except TypeError:
        max_val = None

    for row, i in enumerate(df.index):
        if draw_rows:
            res += f"<tr><th><div class=\"content\">{i}</div></th>"
        for col, j in enumerate(df.columns):
            now = row if transpose else col
            val = convert_num[now](df.loc[i, j])
            if mark_biggest:
                try:
                    actual_val = float(df.loc[i, j])
                except Exception:
                    actual_val = not None
            if row < col or draw_full:
                res += f"<td style=\"background-color: {get_color[now](val)}\">"
                res += "<div class=\"content\">"
                if additional_info is not None:
                    res += f"<div class=\"with-details\">"
                    if mark_biggest and actual_val == max_val:
                        res += "<b>"
                    res += f"{val}"
                    if mark_biggest and actual_val == max_val:
                        res += "</b>"
                    # <actual details>
                    res += "<div class=\"details\">"

                    # yes, I hate magic numbers/code, but use it
                    next_df = additional_info[i][j]
                    next_name = ''
                    if 'cl1' in next_df.columns and 'cl2' in next_df.columns:
                        next_name = f'Кластеры {int(next_df.cl1)} и {int(next_df.cl2)}'
                        next_df = next_df.drop(['cl1', 'cl2'], axis=1)
                    res += csv2html(next_df, drop_noise=False, draw_full=True, name=next_name, square_table=False,
                                    transpose=True, draw_columns=False, mark_biggest=True)
                    res += "</div>"
                    # </actual details>
                    res += "</div>"
                else:
                    if mark_biggest and actual_val == max_val:
                        res += "<b>"
                    res += f"{val}"
                    if mark_biggest and actual_val == max_val:
                        res += "</b>"

                res += "</div>"
                res += "</td>"
            else:
                res += "<div class=\"content\"></div>"
                res += "<td></td>"
        res += "</tr>\n"

    res += """        </table>"""
    return res


def output_img(path_to_img):
    return f"<img src=\"data:image/png;base64,{read_img_as_base64(path_to_img)}\"width=1500 height=1500>"


def read_img_as_base64(path):
    with open(path, 'rb') as image_file:
        bs64 = base64.b64encode(image_file.read())
    return bs64.decode('utf-8')


def color_corr(val):
    if abs(val) < 0.1:
        return 'lime'
    elif abs(val) < 0.3:
        return 'limegreen'
    elif abs(val) < 0.5:
        return 'yellow'
    elif abs(val) < 0.7:
        return 'gold'
    elif abs(val) < 0.8:
        return 'coral'
    elif abs(val) < 0.9:
        return 'tomato'
    else:
        return 'orangered'


def color_cov(val):
    if abs(val) < 0.1:
        return 'lime'
    elif abs(val) < 5:
        return 'limegreen'
    elif abs(val) < 100:
        return 'yellow'
    elif abs(val) < 1000:
        return 'gold'
    elif abs(val) < 2000:
        return 'coral'
    elif abs(val) < 5000:
        return 'tomato'
    else:
        return 'orangered'


def make_csv_corr(sp_names, way_to_an):
    tbl = []
    r_n = {"true_cnt_courses": "кол-во курсов ученика",
           "cnt_courses": "кол-во курсов, с посылками",
           "users_mod_to_all_mod": "доля закрытых модулей ко всем доступным",
           "avg_attempts": "в среднем попыток на задачу",
           "ok_of_tries": "доля вердиктов ок от всех посылок",
           "ok_of_epi": "доля вердиктов ок от всех задач с посылками",
           "ok_to_ct": "доля вердиктов ок от задач в курсах",
           "ok_to_mt": "доля вердиктов ок от задач в открытых модулях",
           "no_tries_to_mt": "доля задач без посылок от всех в открытых модулях",
           "unsuccessful_to_mt": "доля задач с неверными посылками от всех в открытых модулях",
           "nunique_epi_to_ct": "доля задач с попытками от всех задач в курсах",
           "nunique_epi_to_mt": "доля задач с попытками от задач в открытых модулях",
           "tries_to_courses": "доля попыток от числа курсов для человека",
           "mods_after_two_weeks": "Доля модулей, закрытых учеником не ранее, чем за две недели",
           "ended_or_not": "закончил ученик курс или нет",
           "open_modules_to_all_m": "кол-во доступных для решения модулей ко всем",
           "ok_to_mods": "доля кол-ва ок от доступных модулей",
           "no_tries_to_ct": "доля задач из доступных модулей от числа задач в курсах ученика",
           "no_tries_to_mods": "доля задач из доступных модулей без посылок от числа открытых модулей",
           "unsuccessful_to_ct": "доля задач с неудачными попытками от задач в курсах ученика",
           "unsuccessful_to_mods": "доля задач с неудачными попытками от кол-ва открытх модулей",
           "nunique_epi_to_mods": "доля задач с попытками от кол-ва открытых модулей",
           "tries_to_courses": "среднее число попыток на курс",
           "tries_to_mods": "среднее число попыток на модуль"}
    col = []
    for i in range(len(sp_names)):
        col.append(r_n[sp_names[i]])
    for i in range(len(sp_names)):
        tbl.append([0.0] * len(sp_names))
    table = pd.DataFrame(data=tbl, index=col, columns=col)
    an = pd.read_csv(way_to_an)
    del an["Unnamed: 0"]
    for i in range(len(sp_names)):
        for j in range(len(sp_names)):
            if sp_names[i] == sp_names[j]:
                table.at[col[i], col[j]] = -2
            else:
                table.at[col[i], col[j]] = an[sp_names[i]].corr(an[sp_names[j]])
    return table


def make_csv_cov(sp_names, way_to_an):
    tbl = []
    r_n = {"true_cnt_courses": "кол-во курсов ученика",
           "cnt_courses": "кол-во курсов, с посылками",
           "users_mod_to_all_mod": "доля закрытых модулей ко всем доступным",
           "avg_attempts": "в среднем попыток на задачу",
           "ok_of_tries": "доля вердиктов ок от всех посылок",
           "ok_of_epi": "доля вердиктов ок от всех задач с посылками",
           "ok_to_ct": "доля вердиктов ок от задач в курсах",
           "ok_to_mt": "доля вердиктов ок от задач в открытых модулях",
           "no_tries_to_mt": "доля задач без посылок от всех в открытых модулях",
           "unsuccessful_to_mt": "доля задач с неверными посылками от всех в открытых модулях",
           "nunique_epi_to_ct": "доля задач с попытками от всех задач в курсах",
           "nunique_epi_to_mt": "доля задач с попытками от задач в открытых модулях",
           "tries_to_courses": "доля попыток от числа курсов для человека",
           "mods_after_two_weeks": "Доля модулей, закрытых учеником не ранее, чем за две недели",
           "ended_or_not": "закончил ученик курс или нет",
           "open_modules_to_all_m": "кол-во доступных для решения модулей ко всем",
           "ok_to_mods": "доля кол-ва ок от доступных модулей",
           "no_tries_to_ct": "доля задач из доступных модулей от числа задач в курсах ученика",
           "no_tries_to_mods": "доля задач из доступных модулей без посылок от числа открытых модулей",
           "unsuccessful_to_ct": "доля задач с неудачными попытками от задач в курсах ученика",
           "unsuccessful_to_mods": "доля задач с неудачными попытками от кол-ва открытх модулей",
           "nunique_epi_to_mods": "доля задач с попытками от кол-ва открытых модулей",
           "tries_to_courses": "среднее число попыток на курс",
           "tries_to_mods": "среднее число попыток на модуль"}
    col = []
    for i in range(len(sp_names)):
        col.append(r_n[sp_names[i]])
    for i in range(len(sp_names)):
        tbl.append([0.0] * len(sp_names))
    table = pd.DataFrame(data=tbl, index=col, columns=col)
    an = pd.read_csv(way_to_an)
    del an["Unnamed: 0"]
    for i in range(len(sp_names)):
        for j in range(len(sp_names)):
            if sp_names[i] == sp_names[j]:
                table.at[col[i], col[j]] = -2
            else:
                table.at[col[i], col[j]] = cova(an[sp_names[i]], an[sp_names[j]], len(an))
    return table


def cova(x, y, n):
    mean_x, mean_y = sum(x) / n, sum(y) / n
    cov_xy = (sum((x[k] - mean_x) * (y[k] - mean_y) for k in range(n)) / (n - 1))
    return cov_xy


def not_line(sp_names, way_to_an):
    tbl = []
    r_n = {"true_cnt_courses": "кол-во курсов ученика",
           "cnt_courses": "кол-во курсов, с посылками",
           "users_mod_to_all_mod": "доля закрытых модулей ко всем доступным",
           "avg_attempts": "в среднем попыток на задачу",
           "ok_of_tries": "доля вердиктов ок от всех посылок",
           "ok_of_epi": "доля вердиктов ок от всех задач с посылками",
           "ok_to_ct": "доля вердиктов ок от задач в курсах",
           "ok_to_mt": "доля вердиктов ок от задач в открытых модулях",
           "no_tries_to_mt": "доля задач без посылок от всех в открытых модулях",
           "unsuccessful_to_mt": "доля задач с неверными посылками от всех в открытых модулях",
           "nunique_epi_to_ct": "доля задач с попытками от всех задач в курсах",
           "nunique_epi_to_mt": "доля задач с попытками от задач в открытых модулях",
           "tries_to_courses": "доля попыток от числа курсов для человека",
           "mods_after_two_weeks": "Доля модулей, закрытых учеником не ранее, чем за две недели",
           "ended_or_not": "закончил ученик курс или нет",
           "open_modules_to_all_m": "кол-во доступных для решения модулей ко всем",
           "ok_to_mods": "доля кол-ва ок от доступных модулей",
           "no_tries_to_ct": "доля задач из доступных модулей от числа задач в курсах ученика",
           "no_tries_to_mods": "доля задач из доступных модулей без посылок от числа открытых модулей",
           "unsuccessful_to_ct": "доля задач с неудачными попытками от задач в курсах ученика",
           "unsuccessful_to_mods": "доля задач с неудачными попытками от кол-ва открытх модулей",
           "nunique_epi_to_mods": "доля задач с попытками от кол-ва открытых модулей",
           "tries_to_courses": "среднее число попыток на курс",
           "tries_to_mods": "среднее число попыток на модуль"}
    col = []
    for i in range(len(sp_names)):
        col.append(r_n[sp_names[i]])
    for i in range(len(sp_names)):
        tbl.append([0.0] * len(sp_names))
    table = pd.DataFrame(data=tbl, index=col, columns=col)
    an = pd.read_csv(way_to_an)
    del an["Unnamed: 0"]
    return table


def plot(sp_names, way_to_an):
    r_n = {"true_cnt_courses": "кол-во курсов ученика",
           "cnt_courses": "кол-во курсов, с посылками",
           "users_mod_to_all_mod": "доля закрытых модулей ко всем доступным",
           "avg_attempts": "в среднем попыток на задачу",
           "ok_of_tries": "доля вердиктов ок от всех посылок",
           "ok_of_epi": "доля вердиктов ок от всех задач с посылками",
           "ok_to_ct": "доля вердиктов ок от задач в курсах",
           "ok_to_mt": "доля вердиктов ок от задач в открытых модулях",
           "no_tries_to_mt": "доля задач без посылок от всех в открытых модулях",
           "unsuccessful_to_mt": "доля задач с неверными посылками от всех в открытых модулях",
           "nunique_epi_to_ct": "доля задач с попытками от всех задач в курсах",
           "nunique_epi_to_mt": "доля задач с попытками от задач в открытых модулях",
           "tries_to_courses": "доля попыток от числа курсов для человека",
           "mods_after_two_weeks": "Доля модулей, закрытых учеником не ранее, чем за две недели",
           "ended_or_not": "закончил ученик курс или нет",
           "open_modules_to_all_m": "кол-во доступных для решения модулей ко всем",
           "ok_to_mods": "доля кол-ва ок от доступных модулей",
           "no_tries_to_ct": "доля задач из доступных модулей от числа задач в курсах ученика",
           "no_tries_to_mods": "доля задач из доступных модулей без посылок от числа открытых модулей",
           "unsuccessful_to_ct": "доля задач с неудачными попытками от задач в курсах ученика",
           "unsuccessful_to_mods": "доля задач с неудачными попытками от кол-ва открытх модулей",
           "nunique_epi_to_mods": "доля задач с попытками от кол-ва открытых модулей",
           "tries_to_courses": "среднее число попыток на курс",
           "tries_to_mods": "среднее число попыток на модуль"}
    col = []
    tbl = []
    an = pd.read_csv(way_to_an)
    del an["Unnamed: 0"]
    for i in range(len(sp_names)):
        col.append(r_n[sp_names[i]])
    for i in range(len(an)):
        tbl.append([0.0] * len(sp_names))
    r = 0
    table = pd.DataFrame(data=tbl, index=an.index, columns=col)
    for i in range(len(sp_names)):
        for j in range(len(an)):
            table[col[i]][j] = an[sp_names[i]][j]
        sns_plot = sns.pairplot(table)
    return sns_plot.savefig("output.png")


def f(sp, way_to_fail):
    way_to_plot = r"C:\Users\maksi\output.png"
    fail = make_csv_corr(sp, way_to_fail)
    st = """<!DOCTYPE html>
<html>
    <head>
        <title>otchet</title>
        <style>
            table, td, th {
                border: 0.5px solid black;
            }

            table {
                border-collapse: collapse;
            }

            td {
                text-align: center;
                vertical-align: middle;
                font-size: 12px;
            }

            .square_table > tbody > tr > th > div.content {
                min-width: 200px;
                min-height: 200px;
                max-width: 200px;
                max-height: 200px;
                font-size: 25px;
                display: flex;
                justify-content: center;
                align-items: center;
            }

            .square_table div.details {
                display: none;
                border: 1px solid lightgray;
                background-color: #FFFFFF;
                padding: 3px 5px;
            }

            .square_table div.with-details:hover div.details {
                position: absolute;
                display: block;
            }

            .details th {
                padding: 5px;
            }

            table:not(.square_table) th {
                padding: 5px;
            }
        </style>

    </head>
    <body>"""
    plot(sp, way_to_fail)
    st += csv2html(fail, name='Линейная корреляция', get_color=color_corr)
    fail1 = make_csv_cov(sp, way_to_fail)
    st += csv2html(fail1, name='Ковариация',
                   get_color=color_cov) + '<p><lang=ru>Ниже представлены графики pairplot для некоторых наборов данных.</p>' + output_img(
        way_to_plot)
    fail2 = not_line(sp, way_to_fail)
    Html_file = open("otchet.html", "w")
    Html_file.write(st)
    Html_file.close()

#Example of calling function
#You give to function f() list of columns names and full way to your database
#f(
# ['open_modules_to_all_m',
# 'ok_of_tries',
# 'ok_of_epi',
# 'ok_to_mt',
# 'no_tries_to_mt',
# 'unsuccessful_to_mt',
# 'nunique_epi_to_mt',
# 'avg_attempts',
# 'tries_to_mods'],
# r'C:\Users\maksi\ling.csv')
