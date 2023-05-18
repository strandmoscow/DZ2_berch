import csv
import numpy as np
import matplotlib.pyplot as plt
from Latex.latex import make_latex
from mc.markov_clustering import clusterisation

var = 10
group = "РК6-84б"
n = 100
name = "Осипов Арсений Константинович"
name_short = "Карачков Д. С."

states = [i + 1 for i in range(15)]
eps = 0.0000000000000001


def reader(var):
    m = np.zeros((0, len(states)))
    i = len(states)
    # CSV reading and storing info int m matrix
    with open('input/Task2.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='\n')
        for row in spamreader:
            if i < len(states):
                i = i + 1
                m = np.vstack([m, row])
            if row:
                if row[0] == f"#{var}":
                    i = 0
        m = m.astype(float)
        for row in m:
            while sum(row) < 1.0:
                print(f"{row} - {sum(row)} +")
                for i in range(len(row)):
                    if row[i] > 0 and sum(row) < 1.0:
                        row[i] = row[i] + 0.01
            while sum(row) > 1.0:
                print(f"{row} - {sum(row)} -")
                for i in range(len(row)):
                    if row[i] > 0 and sum(row) > 1.0:
                        row[i] = row[i] - 0.01
    return m.astype(float)


def cluster_ordering(clusters, mat):
    ret = []
    nesush = 0
    for key in clusters.keys():
        for item in clusters[key]:
            for i in range(1, len(mat[0])):
                if (mat[item - 1][i] > 0) and (i not in clusters[key]):
                    nesush = key

    ret.append(clusters[nesush])
    for key in clusters.keys():
        if key != nesush:
            ret.append(clusters[key])

    return ret


# чтоб потом загнать в рисовщик графов
def print_matrix(m):
    s = ''
    for i in range(len(m)):
        for j in range(len(m[i])):
            s += str(np.round(m[i][j], 2))
            s += ' '
        s += '\n'
    print(s)


# свертка
def svertka(P, not_such, such):
    n = len(not_such) + len(such)
    svert_matrix = np.zeros((n, n))
    # заполнение сектора для несущественного кластера
    for i in range(n):
        if i < len(not_such):
            v = list(P[i][0:len(not_such)]) + [0 for _ in range(len(such))]
        else:
            v = np.zeros((1, n))
        svert_matrix[i] = v
    # print(svert_matrix)

    # занесение свернутых существенных секторов
    for i in range(len(such)):
        svert_matrix[len(not_such) + i][len(not_such) + i] = 1
    # print(svert_matrix)

    # учет переходов в существенные состояния
    for i in range(len(not_such)):
        p = np.sum(P[i][len(not_such)::])
        if p:
            indx = list(P[i][len(not_such)::]).index(p) + len(not_such) + 1
            d = 1
            for l in such:
                if indx in l:
                    break
                d += 1
            svert_matrix[i][len(such) + d] = p
    return svert_matrix


# получение предельной матрицы
def P_n(P_svert, n):
    s = 1
    while s > eps:
        P_svert = P_svert.dot(P_svert)
        s = 0
        for j in range(n):
            s += np.sum(P_svert[j][0:n])
    return P_svert


def p_vector(pi, Pn, not_such, such, locals_p):
    Pn = Pn.T
    p = Pn.dot(pi)

    n = len(not_such)

    ans = [0 for _ in not_such]
    for i in range(len(such)):
        p_vec = p[n + i] * locals_p[i]
        # ans += [np.round(x, 2) for x in p_vec]
        ans += [x for x in p_vec]
    return ans


# сортировка по кластерам
def change_matrix(pos, P, cltrs):
    n = len(P)
    sort_matrix = np.zeros((n, n))
    for i in range(len(P)):
        sort_matrix[i] = P[pos[i] - 1]

    sort_matrix = sort_matrix.T
    for i in range(len(sort_matrix)):
        P[i] = sort_matrix[i]
    for i in range(len(P)):
        sort_matrix[i] = P[pos[i] - 1]

    # переопределение кластеров
    new_clasters = []
    last_indx = 0
    for claster in cltrs:
        new_clasters.append(states[last_indx:last_indx + len(claster)])
        last_indx += len(claster)
    return sort_matrix.T, new_clasters


# получение локальной матрицы переходов по списку вершин
def make_class_matrix(class_states, P):
    n = len(class_states)
    class_matrix = np.zeros((n, n))
    k = 0

    for i in np.sort(class_states):
        row_prob = []
        for indx in range(len(P[i - 1])):
            if indx + 1 in class_states:
                row_prob.append(P[i - 1][indx])
        class_matrix[k] = row_prob
        k += 1
    return class_matrix


def mark_iter(n, m, states, s_start):
    current_s = s_start
    states_tr = [current_s]
    n_entry = [0 for _ in range(len(states))]
    for _ in range(n - 1):
        per_ver = m[current_s - 1]
        n_entry[current_s - 1] += 1
        next_s = np.random.choice(states, p=per_ver)
        current_s = next_s
        states_tr.append(current_s)
    return n_entry, states_tr


def marginal_probabilities(P):
    A = P.T - np.eye(len(P), dtype=float)
    A[-1] = np.full(len(P), 1)
    b = np.zeros(len(P))
    b[-1] = 1
    p = np.linalg.solve(A, b)
    return p


def the_third_task(vertex, P):
    local_P = make_class_matrix(vertex, P)
    local_p_vector = marginal_probabilities(local_P)
    return local_p_vector, local_P


def the_forth_task(clusters, P, locs_p):
    # свертка с поиском предельной
    P = svertka(P, clusters[0], clusters[1::])
    Pn = P_n(P, len(clusters[0]))
    # print_matrix(P)

    # 4.1
    ans1 = []
    for i in range(len(clusters[0])):
        pi = [0 for _ in range(len(clusters[0]) + len(clusters[1::]))]
        pi[i] = 1
        ans1_1 = p_vector(pi, Pn, clusters[0], clusters[1::], locs_p)
        ans1.append(ans1_1)
        # print(ans)
        # print(np.sum(ans))

    # 4.2
    pi = [1 / len(clusters[0]) for _ in range(len(clusters[0]))] + [0 for _ in clusters[1::]]
    ans2 = p_vector(pi, Pn, clusters[0], clusters[1::], locs_p)
    # print(ans)
    # print(np.sum(ans))

    # P - свернутая матрица, Pn - предельная матрица переходов,
    # ans1 - набор векторов предельных вероятностей для пункта 4.1
    # ans2 - вектор предельных вероятностей для пункта 4.2
    return P, Pn, ans1, ans2


def matr_to_table_r_gr(mat, n, delim):
    s = ""
    for i in mat:
        for x in i:
            s += f'{round(x, n)}'
            s += f'{delim}'
        s = s[:-len(delim)]
        s = s + " \n"
    s = s[:-3]
    return s


if __name__ == "__main__":
    m = reader(var)
    m_classic = reader(var)

    print("\nКласическая матрица:")
    print(matr_to_table_r_gr(m_classic, 3, ", "))
    print("\n")

    with open("input/Task2_ed.csv", mode="w+", encoding="utf-8") as base:
        base.write(matr_to_table_r_gr(m_classic, 2, ", "))
        print(f"... wrote input/Task2_ed.csv")

    # clusters = cluster_ordering(clusterisation("input/Task2_ed.csv"), m)
    clusters = [[2, 3, 5, 13], [2, 4, 11, 14], [8, 9, 10], [6, 7, 12, 15]]

    for line in m_classic:
        print(sum(line))

    print(clusters)

    # 3 задание
    vertex_list = []
    vertex_list += clusters[0]
    locals_p = []
    rgo = []
    for claster in clusters[1::]:
        p, rgo_1 = the_third_task(claster, m)
        locals_p.append(p)
        rgo.append(rgo_1)
        vertex_list += claster

    # сортировка по кластерам
    m_svert, clusters = change_matrix(vertex_list, m, clusters)
    # print(clusters)

    P, Pn, ans1, ans2 = the_forth_task(clusters, m_svert, locals_p)

    # отрисовка переходов
    plt.grid()
    plt.ylabel("Состояния")
    plt.xlabel("Переходы")
    plt.yticks(np.arange(0, len(states) + 1, step=1))

    steps = 10
    x = [i + 1 for i in range(n)]

    list_of_last_states = [0] * 15

    for i in range(steps):
        for j in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
            myDictionary, tr = mark_iter(n, m_svert, states, j)
            plt.plot(x, tr, 'o--', linewidth=0.5, markeredgewidth=0.1)
            list_of_last_states[tr[-1] - 1] += 1
    plt.savefig('Latex/res/Images/iter.png')

    make_latex("{" + f"{var}" + "}", "{" + f"{group}" + "}", "{" + f"{name}" + "}", "{" + f"{name_short}" + "}",
               m_classic, rgo, locals_p, m_svert, P, Pn, ans1, ans2, list_of_last_states)
