import pandas as pd
import numpy as np
from collections import defaultdict
import tqdm
from sklearn.neighbors import KDTree
import base64
from copy import deepcopy
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from numpy import linalg
import plotly.express as px
import uuid
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

RED = [255, 0, 0]
ORANGE = [255, 135, 0]
GREEN = [75, 255, 0]
PLUM = [255, 230, 255]
BLUE = [210, 255, 255]
CONVERTING = {}
DELETED = set()
COLORS3 = ["#9d6d00", "#903ee0", "#11dc79", "#f568ff", "#419500", "#013fb0",
           "#f2b64c", "#007ae4", "#ff905a", "#33d3e3", "#9e003a", "#019085",
           "#950065", "#afc98f", "#ff9bfa", "#83221d", "#01668a", "#ff7c7c",
           "#643561", "#75608a"]

WORDS = {}


def get_dictionary(path):
    if not path:
        return {}
    return json.load(open(path, encoding='utf-8'))


def lda_fit_and_transform(data_, labels):
    scaler = StandardScaler()
    data = scaler.fit_transform(data_)

    labels_to_data = {k: [] for k in labels}

    for row, label in zip(data, labels):
        labels_to_data[label].append(row)

    for k, v in labels_to_data.items():
        labels_to_data[k] = np.array(v)

    means = {}

    for k, v in labels_to_data.items():
        means[k] = v.mean(axis=0)

    total_mean = data.mean(axis=0)

    sw = None

    for k, v in labels_to_data.items():
        vects = v - means[k]
        mat = np.dot(vects.T, vects)

        if sw is None:
            sw = mat
        else:
            sw += mat

    sb = None

    for k, v in labels_to_data.items():
        vect = np.array([means[k] - total_mean])
        mat = len(v) * np.dot(vect.T, vect)

        if sb is not None:
            sb += mat
        else:
            sb = mat

    s = np.dot(linalg.inv(sw), sb)
    w, v = linalg.eig(s)

    [(eigv1_ind, _), (eigv2_ind, _)] = list(sorted(enumerate(w), key=lambda x: -np.abs(x[1])))[:2]

    W = np.real(np.array([v[:, eigv1_ind], v[:, eigv2_ind]]))
    res = np.dot(data, W.T)

    return res.astype('float64')


def calc_euclidean_dist(a: np.array, b: np.array):
    return np.linalg.norm(a - b)


def normalize_data(data):
    vals = []
    for i in data.columns:
        # standard scaler
        #         vals.append([])
        #         vals[-1].append(np.mean(data[i]))
        #         data[i] -= vals[-1][-1]
        #         vals[-1].append(np.std(data[i]))
        #         data[i] /= vals[-1][-1]

        # minmax scaler
        vals.append([np.min(data[i])])
        data[i] -= np.min(data[i])
        vals[-1].append(np.max(data[i]))
        data[i] /= np.max(data[i])
    return data, np.array(vals)


def unormalize_data(data, vals):
    assert data.shape[1] == vals.shape[0]
    data1 = data.copy()
    now = 0
    for i in data.columns:
        data1[i] *= vals[now][1]
        data1[i] += vals[now][0]
        now += 1
    return data1


def split_by_cluster(data: pd.DataFrame) -> dict:
    drop_percent = ['percent'] if 'percent' in data.columns else []

    di = defaultdict(list)
    for i in data.reset_index().drop(drop_percent, axis=1).itertuples():
        now = i._asdict()
        del now['Index'], now['cluster'], now['user_id']
        di[i.cluster].append(now)
    return di


def get_mean_vectors(di: dict) -> dict:
    base = dict()
    for key, values in di.items():
        now = None
        for j in values:
            if now is not None:
                now += np.array(list(j.values()))
            else:
                now = np.array(list(j.values()))
        if now is None:
            continue
        now /= len(values)
        base[key] = np.array(now)
    return base


def get_distances(data: pd.DataFrame, base: dict) -> dict:
    distances = defaultdict(list)
    for i in data.reset_index().itertuples():
        now = i._asdict()
        cluster = now['cluster']
        del now['Index'], now['cluster']
        user_id = now['user_id']
        del now['user_id']
        now = np.array(list(now.values()))
        distances[i.cluster].append((user_id, calc_euclidean_dist(base[cluster], now)))
    return distances


def get_cluster_centers(data: pd.DataFrame, filename):
    # split by cluster
    di = split_by_cluster(data)
    # get mean vectors
    base = get_mean_vectors(di)
    # get distances
    distances = get_distances(data, base)
    # get points
    res = defaultdict(list)
    for i in distances:
        distances[i].sort(key=lambda x: x[1])
    for i in distances:
        now = []
        for num, j in enumerate(distances[i]):
            now.append(j[0])
            if num >= 3:
                break
        res['cluster_title'].append(i)
        res['cluster_points'].append('[' + ', '.join(map(str, now)) + ']')

    pd.DataFrame(res).to_csv(filename)


def get_circles(data: pd.DataFrame):
    data.cluster = data.cluster.astype('int')
    # split by cluster
    di = split_by_cluster(data)
    # get center points
    base = get_mean_vectors(di)
    # get distances
    distances = get_distances(data, base)
    # get the farthest
    for i in distances:
        distances[i].sort(key=lambda x: -x[1])
        distances[i] = distances[i][0][1]
    # get circles
    circles = dict()
    for i in distances:
        circles[i] = (base[i], distances[i])
    return circles


def calc_circles_percent(data: pd.DataFrame):
    data.cluster = data.cluster.astype('int')
    circles = get_circles(data)
    columns = np.array(list(circles.keys()))
    columns.sort()
    res = defaultdict(dict)
    for i in columns:
        c1, r1 = circles[i]
        for j in columns:
            c2, r2 = circles[j]
            d1, d2 = np.array(data[data.cluster == i].drop('cluster', axis=1)), \
                     np.array(data[data.cluster == j].drop('cluster', axis=1))
            all_len = d1.shape[0] + d2.shape[0]
            d = np.concatenate((d1, d2))
            res[i][j] = \
                d[(np.sum((d - c1) ** 2, axis=1) <= r1 ** 2) & (np.sum((d - c2) ** 2, axis=1) <= r2 ** 2)].shape[
                    0] / all_len
    return pd.DataFrame(res, columns=columns, index=columns)


def calc_centers_dist(data: pd.DataFrame):
    data.cluster = data.cluster.astype('int')

    circles = get_circles(data)
    cls = np.array(data.columns.drop('cluster'))
    columns = np.array(list(circles.keys()))
    columns.sort()
    res = defaultdict(dict)
    centers_delta = defaultdict(dict)
    for i in columns:
        c1, r1 = circles[i]
        for j in columns:
            c2, r2 = circles[j]
            dist = calc_euclidean_dist(c1, c2)
            res[i][j] = dist

            new_df = pd.DataFrame(((c1 - c2) ** 2).reshape(1, len(cls)), columns=cls)
            values = np.array(new_df)[0]
            values = list(zip(new_df.columns, values))
            # values.sort(key=lambda x: x[1], reverse=True)
            new_cls = np.array([i[0] for i in values] + ['cl1', 'cl2'])
            values = np.array([i[1] for i in values] + [int(i), int(j)])
            values = values.reshape(1, values.shape[0])
            centers_delta[i][j] = pd.DataFrame(values, columns=new_cls)
            del new_df

    return pd.DataFrame(res, columns=columns, index=columns), centers_delta


def calc_dist_circle_centers(data: pd.DataFrame):
    data.cluster = data.cluster.astype('int')
    circles = get_circles(data)

    columns = np.array(list(circles.keys()))
    columns.sort()
    cls = np.array(data.columns.drop('cluster'))
    res = defaultdict(dict)
    additional_info = defaultdict(dict)
    for i in columns:
        c1, r1 = circles[i]
        for j in columns:
            c2, r2 = circles[j]
            dist = calc_euclidean_dist(c1, c2)
            if dist > r1 + r2:
                res[i][j] = dist - r1 - r2
            elif dist < r1 and dist < r2:
                res[i][j] = -dist
            else:
                res[i][j] = dist - r1 - r2

            new_df = pd.DataFrame(((c1 - c2) ** 2).reshape(1, len(cls)), columns=cls)

            values = np.array(new_df)[0]
            values = list(zip(new_df.columns, values))
            # values.sort(key=lambda x: x[1], reverse=True)
            new_cls = np.array([i[0] for i in values] + ['cl1', 'cl2'])
            values = np.array([i[1] for i in values] + [int(i), int(j)])
            values = values.reshape(1, values.shape[0])
            additional_info[i][j] = pd.DataFrame(values, columns=new_cls)
            del new_df

    return pd.DataFrame(res, columns=columns, index=columns), additional_info


def calc_nearest_dist(data: pd.DataFrame):
    trees = dict()
    for i in data.cluster.unique():
        need_data = data[data.cluster == i]
        trees[i] = KDTree(np.array(need_data.drop('cluster', axis=1)))

    data.cluster = data.cluster.astype('int')
    columns = sorted(data.cluster.unique())
    distances = defaultdict(dict)
    mins = defaultdict(dict)
    cls = data.drop('cluster', axis=1).columns
    for cl1 in columns:
        for cl2 in columns:
            data2 = np.array(data[data.cluster == cl2].drop('cluster', axis=1))
            dist, ind = trees[cl1].query(data2, k=1)
            dist = dist[:, 0]
            ind = ind[:, 0]
            distances[cl1][cl2] = dist.min()

            t1, t2 = data[data.cluster == cl1].iloc[ind[dist.argmin()]][:-1], data[data.cluster == cl2].iloc[
                                                                                  dist.argmin()][:-1]
            new_df = pd.DataFrame(np.array((t1 - t2) ** 2).reshape(1, len(cls)), columns=cls)

            values = np.array(new_df)[0]
            values = list(zip(new_df.columns, values))
            # values.sort(key=lambda x: x[1], reverse=True)
            new_cls = np.array([i[0] for i in values] + ['cl1', 'cl2'])
            values = np.array([i[1] for i in values] + [int(cl1), int(cl2)])
            values = values.reshape(1, values.shape[0])
            mins[cl1][cl2] = pd.DataFrame(values, columns=new_cls)
            del new_df

    df = pd.DataFrame(distances, columns=columns, index=columns)
    return df, mins


def get_cluster_sizes(data: pd.DataFrame):
    cl_names = sorted(pd.unique(data['cluster']))
    cl_keys = dict()
    for i in range(len(cl_names)):
        cl_keys[cl_names[i]] = i
    cl_sizes = [0] * len(cl_names)

    for i in data['cluster']:
        col = cl_keys[i]
        cl_sizes[col] += 1

    my_table = []
    for i in cl_sizes:
        my_table.append([i, i / data.shape[0]])

    df = pd.DataFrame(my_table, columns=['size', 'part'], index=cl_names)
    return df


def mkchrs(data: pd.DataFrame, data_unscaled: pd.DataFrame):
    vectorized_square_distance = np.vectorize(lambda x, y: (x - y) ** 2, otypes=[float])
    kd = KernelDensity()
    l, l_unscaled = dict(), dict()
    for x in data.cluster.unique():
        l[x] = data.query(f'cluster=={x}').drop(["cluster"], axis=1).values
        l_unscaled[x] = data_unscaled[data_unscaled.cluster == x].drop('cluster', axis=1).values
    mean, std, density, diameter = deepcopy(l_unscaled), deepcopy(l_unscaled), deepcopy(l), deepcopy(l)
    for x in mean.keys():
        mean[x] = mean[x].mean(axis=0)
    for x in std.keys():
        std[x] = std[x].std(axis=0)
    mean = pd.DataFrame(mean).transpose()
    std = pd.DataFrame(std).transpose()
    ans = mean.astype(str)
    for i in ans.columns:
        for j in data.cluster.unique():
            ans.at[j, i] = '{:.2f} ± {:.2f}'.format(mean.at[j, i], 3 * std.at[j, i])
    ans = ans.rename(lambda x: data.columns[x], axis=1)
    for x in density.keys():
        kd.fit(density[x])
        density[x] = convert_float(-kd.score(density[x]))
    # for x, y in diameter.items():
    #     max_square_distance = 0
    #     n = y.shape[0]
    #     for i in tqdm.tqdm(range(n)):
    #         vals = vectorized_square_distance(y[i:], y[i]).sum(axis=1).max()
    #         max_square_distance = max(max_square_distance, vals)
    #     diameter[x] = max_square_distance

    density = pd.Series(density)
    # diameter = pd.Series(diameter)
    ans = pd.DataFrame(density.rename('density')).join(ans)
    # ans = pd.DataFrame(diameter.rename('diameter')).join(ans)
    ans = ans.reset_index().rename(columns={'index': 'cluster'}).set_index('cluster')
    return ans


def merge_tables(nearest_points: pd.DataFrame, centers_dist: pd.DataFrame, circles_dist: pd.DataFrame,
                 circles_percent: pd.DataFrame):
    add = defaultdict(dict)
    res = defaultdict(dict)
    columns = nearest_points.columns
    for i in columns:
        for j in columns:
            now_data = np.array([[nearest_points.loc[i, j], centers_dist.loc[i, j], circles_dist.loc[i, j],
                                  circles_percent.loc[i, j], i, j]])
            now = pd.DataFrame(now_data, columns=['Минимальное расстоние', 'Расстояние между центрами',
                                                  'Расстояния между сферами', 'Доля точек в пересечении', 'cl1', 'cl2'])
            add[i][j] = now
            res[i][j] = [nearest_points.loc[i, j], centers_dist.loc[i, j], circles_dist.loc[i, j],
                         circles_percent.loc[i, j]]
    return pd.DataFrame(res, index=columns, columns=columns), pd.DataFrame(add)


def convert_float(num):
    return round(float(num), 2)


def no_color(color):
    return "#ffffff"


def csv2html(df: pd.DataFrame, get_color=no_color, drop_noise=True, name='', additional_info=None,
             draw_full=False, convert_num=convert_float, square_table=True, transpose=False,
             draw_columns=True, draw_rows=True, mark_biggest=False, use_actual_val=False,
             func_mark=lambda x: np.array(x).max()):
    if isinstance(convert_num, list) or isinstance(convert_num, tuple):
        pass
    else:
        convert_num = [convert_num for _ in range(df.shape[1])]

    if isinstance(get_color, list) or isinstance(get_color, tuple):
        pass
    else:
        get_color = [get_color for _ in range(df.shape[1])]

    sq_class = "class=\"square_table\"" if square_table else ""
    res = ""
    if name:
        res += f"<h4>{name}</h4>"
    res += f"""<table {sq_class}>"""

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
            res += f"<th><div class=\"content\">{WORDS.get(i, i)}</div></th>"
        res += "</tr>\n"
    #
    try:
        max_val = func_mark(df)
    except TypeError:
        max_val = None

    for row, i in enumerate(df.index):
        if draw_rows:
            res += f"<tr><th><div class=\"content\">{WORDS.get(i, i)}</div></th>"
        for col, j in enumerate(df.columns):
            now = row if transpose else col
            val = convert_num[now](df.loc[i, j])
            try:
                actual_val = df.loc[i, j]
            except Exception:
                actual_val = not None
            if row < col or draw_full:
                res += f"<td style=\"background-color: {get_color[now](actual_val if use_actual_val else val)}\">"
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
                                    transpose=True, draw_columns=False, mark_biggest=True, func_mark=func_mark,
                                    get_color=color_zero)
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


def parse_color(color: str):
    color = color[1:]
    res = []
    for i in [0, 2, 4]:
        res.append(int(color[i:i + 2], 16))
    return res


def color_zero(color):
    if color == 0:
        return 'lightgray'
    else:
        return None


def color_merged_tables(color):
    nearest, centers, circles, percent = (color[0], parse_color(color_nearest_dist(color[0]))), \
                                         (color[1], parse_color(color_centers_dist(color[1]))), (
                                             color[2], parse_color(color_dist_circle(color[2]))), (
                                             color[3], parse_color(color_circle_percent(color[3])))
    need = min(nearest, centers, circles)[1]
    return '#' + ''.join(['0' * (len(hex(i)[2:]) == 1) + hex(i)[2:] for i in need])


def color_circle_percent(color):
    if color <= 0.05:
        new_color = [GREEN[i] for i in range(3)]
    elif color <= 0.4:
        w1 = (color - 0.05) / (0.4 - 0.05)
        w2 = 1 - w1
        new_color = [round(ORANGE[i] * w1 + GREEN[i] * w2) for i in range(3)]
    else:
        w2 = (color - 0.4) / (1 - 0.4)
        w1 = 1 - w2
        new_color = [round(RED[i] * w2 + ORANGE[i] * w1) for i in range(3)]
    return '#' + ''.join(['0' * (len(hex(i)[2:]) == 1) + hex(i)[2:] for i in new_color])


def color_dist_circle(color):
    # RGB
    if color <= -0.3:
        new_color = [RED[i] for i in range(3)]
    elif color <= -0.05:
        w1 = (color - -0.05) / -0.25
        w2 = 1 - w1
        new_color = [round(RED[i] * w1 + ORANGE[i] * w2) for i in range(3)]
    else:
        new_color = [round(GREEN[i]) for i in range(3)]
    return '#' + ''.join(['0' * (len(hex(i)[2:]) == 1) + hex(i)[2:] for i in new_color])


def color_centers_dist(color):
    # RGB
    if color < 0.2:
        w2 = color / 0.2
        w1 = 1 - w2
        new_color = [round(RED[i] * w1 + ORANGE[i] * w2) for i in range(3)]
    elif color < 0.35:
        w2 = (color - 0.2) / (0.35 - 0.2)
        w1 = 1 - w2
        new_color = [round(ORANGE[i] * w1 + GREEN[i] * w2) for i in range(3)]
    else:
        new_color = [round(GREEN[i]) for i in range(3)]
    return '#' + ''.join(['0' * (len(hex(i)[2:]) == 1) + hex(i)[2:] for i in new_color])


def color_nearest_dist(color):
    # RGB
    if color < 0.1:
        w2 = color / 0.1
        w1 = 1 - w2
        new_color = [round(RED[i] * w1 + ORANGE[i] * w2) for i in range(3)]
    elif color < 0.25:
        w2 = (color - 0.1) / (0.25 - 0.1)
        w1 = 1 - w2
        new_color = [round(ORANGE[i] * w1 + GREEN[i] * w2) for i in range(3)]
    else:
        new_color = [round(GREEN[i]) for i in range(3)]
    return '#' + ''.join(['0' * (len(hex(i)[2:]) == 1) + hex(i)[2:] for i in new_color])


def color_cl_sizes(color):
    if isinstance(color, str):
        return None
    # RGB
    if color < 0.05:
        w2 = color / 0.05
        w1 = 1 - w2
        new_color = [round(PLUM[i] * w1 + BLUE[i] * w2) for i in range(3)]
        # new_color = [255, 255, 255]
    elif color < 0.25:
        new_color = GREEN
    elif color > 0.5:
        new_color = RED
    elif 0.25 <= color < 0.35:
        w2 = (color - 0.25) / (0.35 - 0.25)
        w1 = 1 - w2
        new_color = [round(GREEN[i] * w1 + ORANGE[i] * w2) for i in range(3)]
    else:
        w2 = (color - 0.35) / (0.5 - 0.35)
        w1 = 1 - w2
        new_color = [round(ORANGE[i] * w1 + RED[i] * w2) for i in range(3)]
    return '#' + ''.join(['0' * (len(hex(i)[2:]) == 1) + hex(i)[2:] for i in new_color])


def get_plot_filename(common_path):
    now_path = f'{common_path}/plots/'
    now_file = uuid.uuid4().hex
    now_path += now_file + '.png'
    os.makedirs(f'{common_path}/plots/', exist_ok=True)
    return now_path


def get_lda(df_unscaled, common_path):
    now_df = df_unscaled.copy()
    now_df.cluster = now_df.cluster.astype('str')
    # delete shit
    for i in now_df.columns:
        if (now_df[i].nunique() == 1 or str(i).endswith('.1')) and i != "cluster":
            now_df = now_df.drop(i, axis=1)
    df_transformed = lda_fit_and_transform(now_df.drop('cluster', axis=1), now_df.cluster)
    now_colors = dict(zip(sorted(now_df.cluster.unique()), COLORS3[:now_df.cluster.nunique()]))
    print(dict(now_colors))
    fig1 = px.scatter(x=df_transformed[:, 0],
                      y=df_transformed[:, 1],
                      color=now_df.cluster,
                      color_discrete_map=now_colors)
    fig1.update_traces(marker=dict(size=12,
                                   line=dict(width=1)),
                       selector=dict(mode='markers'))
    fig1.update()
    now_path1 = get_plot_filename(common_path)
    fig1.write_image(now_path1)
    res = f"""
        <table class="imgtable" cellspacing="0" cellpadding="0">
        <tr><th>Без шумов</th></tr>
        <tr><td><img  src=\"data:image/png;base64,{read_img_as_base64(now_path1)}\"></td></tr>
        </table>
    """
    return res


def get_pairplot(df_unscaled, common_path):
    now_df = df_unscaled.copy()
    now_df.cluster = now_df.cluster.astype('str')
    now_df.columns = list(map(lambda x: WORDS.get(x, x), now_df.columns))
    now_df = now_df[now_df.cluster != "-1"]
    now_colors = dict(zip(sorted(now_df.cluster.unique()), COLORS3[:now_df.cluster.nunique()]))
    print(dict(now_colors))
    plot = sns.pairplot(now_df, hue='cluster', palette=now_colors)
    now_path = get_plot_filename(common_path)
    plot.savefig(now_path)
    plt.clf()
    return f"<img src=\"data:image/png;base64,{read_img_as_base64(now_path)}\"><br>"


def get_df_paths(common_path, scaled_table_name, unscale_table_name):
    source_paths = [os.path.join(common_path, scaled_table_name),
                    os.path.join(common_path, unscale_table_name)]
    return source_paths


def get_df(common_path, scaled_table_name, unscale_table_name):
    paths = get_df_paths(common_path, scaled_table_name, unscale_table_name)
    scaled = pd.read_csv(paths[0], index_col=0, sep=',')
    unscaled = pd.read_csv(paths[1], index_col=0, sep=',')
    try:
        scaled = scaled.set_index('user_id')
    except Exception as e:
        print(f"Warning: {e}")

    try:
        unscaled = scaled.set_index('user_id')
    except Exception as e:
        print(f"Warning: {e}")

    return scaled, unscaled


def process_int_numbers(num):
    num = int(num)
    res = list(reversed(str(num)))
    very_res = ''
    now = 0
    for i in res:
        very_res += i
        now += 1
        if now == 3:
            very_res += ' '
            now = 0
    return ''.join(list(reversed(very_res)))


def read_img_as_base64(path):
    if not path:
        return ''
    with open(path, 'rb') as image_file:
        bs64 = base64.b64encode(image_file.read())
    return bs64.decode('utf-8')


def insert_plots(df_unscaled, cluster, common_path, drop_noise=True):
    res = "<table class=\"plot-table\">"
    cr = 6
    now = df_unscaled[df_unscaled.cluster == cluster]
    for num, j in enumerate(now.columns):
        if j == 'cluster':
            continue
        if num % cr == 0:
            res += "<tr>"
        now_plot = sns.kdeplot(data=now, x=j, palette=[COLORS3[int(cluster)]], fill=True, hue='cluster')
        now_path = get_plot_filename(common_path)
        now_plot.get_figure().savefig(now_path)
        now_plot.get_figure().clf()
        res += f"<td><img src=\"data:image/png;base64,{read_img_as_base64(now_path)}\" width=\"199\"" \
               f" height=\"199\"><br><p>{WORDS.get(j, j)}</p>"
        if num % cr == cr - 1:
            res += "</tr>"
    if not res.endswith("</tr>"):
        res += "</tr>"
    res += "</table>"
    res += "<br>"
    return res


def insert_images(common_img_path, add_plots, df_unscaled, common_path, drop_noise=True):
    res = ""
    header = set()
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))
    for cluster in os.listdir(common_img_path):
        if cluster.startswith('.') or cluster.startswith('__'):
            continue
        cur_cluster = int(cluster.split('_')[-1])
        if cur_cluster in DELETED:
            continue
        if drop_noise and cur_cluster == -1:
            continue
        cur_cluster = CONVERTING[cur_cluster]
        for photo in os.listdir(os.path.join(common_img_path, cluster)):
            if photo.startswith('.') or photo.startswith('__'):
                continue
            pupil, course = map(int, os.path.splitext(photo)[0].split('_'))
            data[cur_cluster][pupil][course] = os.path.join(common_img_path, cluster, photo)
            header.add(course)
    res += "<table>"
    header = sorted(list(header))
    res += "<tr>"
    for i in header:
        res += f"<th>{i}</th>"
    res += "</tr>"

    for cl in sorted(data.keys()):
        pupils = data[cl]
        res += f"<tr><th colspan=\"{len(header) + 1}\">Кластер {cl}</td></tr>"
        for pupil, courses in pupils.items():
            now_data = ['' for _ in range(len(header))]
            for course, path in courses.items():
                ind = header.index(int(course))
                now_data[ind] = path
            for i in now_data:
                if not i:
                    res += f"<td></td>"
                else:
                    res += f"<td><img src=\"data:image/png;base64,{read_img_as_base64(i)}\" " \
                           f"width=\"228\" height=\"216\"></td>"
            res += "</tr>"
        if add_plots:
            res += f"<tr><td colspan=\"{len(header) + 1}\">{insert_plots(df_unscaled, cl, common_path)}</td></tr>"
    res += "</table>"
    return res


def visualise_statistics(common_path, scaled_table_name, unscale_table_name, images_folder,
                         name="Visualization", output="statistics.html", add_plots=False):
    global DELETED, CONVERTING
    res = f"""<!DOCTYPE html>
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
        <title>{name}</title>
        <style>
            table, td, th {{
                border: 0.5px solid black;
            }}

            table {{
                border-collapse: collapse;
            }}

            table.imgtable, table.imgtable td, table.imgtable th {{
                border: none;
            }}

            td {{
                text-align: center;
                vertical-align: middle;
                font-size: 12px;
            }}

            .square_table > tbody > tr > th > div.content {{
                min-width: 30x;
                min-height: 30px;
                max-width: 30px;
                max-height: 30px;
                display: flex;
                justify-content: center;
                align-items: center;
            }}

            .square_table > tbody > tr > td, .square_table > tbody > tr > th {{
                width: 30px;
            }}

            .square_table div.details {{
                display: none;
                border: 1px solid lightgray;
                background-color: #FFFFFF;
                padding: 3px 5px;
            }}

            .square_table div.with-details:hover div.details {{
                position: absolute;
                display: block;
            }}

            .details td, .details th {{
                padding: 5px;
            }}

            table:not(.square_table) th {{
                padding: 5px;
            }}

            .details th {{
                text-align: right;
            }}
            
            .plot-table td, .plot-table {{
                border: none;
            }}
        </style>"""

    res += f"""</head>
    <body><h1>{name}</h1>"""

    CONVERTING = dict()
    DELETED = set()
    # получение табличечек
    df, df_unscaled = get_df(common_path, scaled_table_name, unscale_table_name)
    # размерчики классиков
    print('CLUSTER SIZES')
    cluster_sizes = get_cluster_sizes(df)

    # сортировочка и фильтрование
    bad_clusters = pd.Series(cluster_sizes[cluster_sizes.part < 0.01].index)
    DELETED.add(-1)
    for i in bad_clusters:
        DELETED.add(i)
    df = df[~df.cluster.isin(DELETED)]
    df_unscaled = df_unscaled[~df_unscaled.cluster.isin(DELETED)]
    cluster_sizes = cluster_sizes[~cluster_sizes.index.isin(DELETED)]

    to_sort = cluster_sizes.copy()
    to_sort.sort_values(by='size', inplace=True, ascending=False)
    for num, i in enumerate(to_sort.index):
        CONVERTING[i] = num
    df.cluster = df.cluster.map(lambda x: CONVERTING[x])
    df_unscaled.cluster = df_unscaled.cluster.map(lambda x: CONVERTING[x])
    cluster_sizes.index = cluster_sizes.index.map(lambda x: CONVERTING[x])

    # графички, картиночки
    print('LDA')
    res += "<h4>Графики с проекциями данных на оптимальную плоскость</h4>"
    #     return df_unscaled.head()
    res += get_lda(df_unscaled, common_path)
    print('PAIRPLOT')
    res += "<h4>Графики попарных зависимостей между характеристиками</h4>"
    res += get_pairplot(df_unscaled, common_path)
    #     "характеристички ЯМ"
    print('GETTING DATA')
    stats = mkchrs(df, df_unscaled)
    nearest_dist, mins = calc_nearest_dist(df)
    centers_dist, centers_delta = calc_centers_dist(df)
    dist_circle, add_info = calc_dist_circle_centers(df)
    circle_percent = calc_circles_percent(df)

    #     res + "размерчики классиков и характеристички ЯМ"
    print('MKCHRS')
    all_stats = pd.merge(cluster_sizes.reset_index(), stats.reset_index(), left_on='index', right_on='cluster',
                         how='left').drop('cluster', axis=1).set_index('index')
    all_stats.sort_values(by='size', inplace=True, ascending=False)
    res += csv2html(all_stats, name='Размеры и характеристики', draw_full=True,
                    get_color=[no_color, color_cl_sizes] + [no_color] * (stats.shape[1]),
                    convert_num=[process_int_numbers, convert_float, process_int_numbers] + [str] * (
                            stats.shape[1] - 1), square_table=False)
    res += "<br><br>"

    #     res += "сводная инфа"
    print('TOTAL INFO')
    merged, merged_info = merge_tables(nearest_dist, centers_dist, dist_circle, circle_percent)
    res += csv2html(merged, name='Общая информация', convert_num=lambda x: convert_float(np.min(x)),
                    additional_info=merged_info, get_color=color_merged_tables,
                    use_actual_val=True, func_mark=lambda x: np.array(x).min())

    #     res += "расстояния между ближайшими элементиками"
    print('NEAREST INFO')
    res += csv2html(nearest_dist, name='Расстояния между ближайшими точками', get_color=color_nearest_dist,
                    additional_info=mins, convert_num=convert_float)
    res += "<br><br>"

    #     res += "расстояния между центриками"
    print('CENTER INFO')
    res += csv2html(centers_dist, name='Расстояния между центрами', get_color=color_centers_dist,
                    additional_info=centers_delta, convert_num=convert_float)
    res += "<br><br>"

    #     res += "расстояния между окружностями"
    print('CIRCLE INFO')
    res += csv2html(dist_circle, name='Расстояния между описанными сферами', get_color=color_dist_circle,
                    additional_info=add_info, convert_num=convert_float)
    res += "<br><br>"

    #     res += "процент точек внутри окружностей"
    print('PERCENT INFO')
    res += csv2html(circle_percent, name='Доля точек в пересечении', convert_num=convert_float,
                    get_color=color_circle_percent)
    res += "<br><br>"

    #     res + "скриншоты по кластерочкам"
    print('IMAGES')
    res += f"<h4>Скриншоты прохождений курсов</h4>"
    res += insert_images(os.path.join(common_path, images_folder), add_plots, df_unscaled, common_path)

    print('SAVING')
    res += """</body></html>"""
    html_file = open(output, "w", encoding='utf-8')
    html_file.write(res)
    html_file.close()
    print('DONE')


def main():
    global WORDS
    parser = argparse.ArgumentParser()
    parser.add_argument('--common-path', default='', type=str, help='Path to directory with other files')
    parser.add_argument('--scaled-table', type=str, help='Path to scaled table (regarding --common-path)',
                        required=True)
    parser.add_argument('--unscaled-table', type=str, help='Path to unscaled table (regarding --common-path)',
                        required=True)
    parser.add_argument('--images', type=str, help='Path to images folder (regarding --common-path)', required=True)
    parser.add_argument('--title', type=str, default='stats', help='Header and title of resulting file')
    parser.add_argument('-o', type=str, default='stats.html', help='Output file name')
    parser.add_argument('--dictionary', type=str, help='Path to dictionary (not regarding --common-path)', default='')
    parser.add_argument('--add-plots', action="store_true")
    args = parser.parse_args()
    WORDS = get_dictionary(args.dictionary)
    visualise_statistics(args.common_path, args.scaled_table, args.unscaled_table, args.images,
                         args.title, args.o, args.add_plots)


if __name__ == '__main__':
    main()
