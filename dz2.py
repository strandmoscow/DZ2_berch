import csv
import numpy as np
from random import randrange
import matplotlib.pyplot as plt

eps = 0.0000000000000001
var = 82
n = 100
states = [i + 1 for i in range(15)]
clusters = [[5, 7, 8, 15], [1, 9, 14], [2, 10, 12, 13], [3, 4, 6, 11]]


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
    for i in range(n):
        if i < len(not_such):
            v = P[i][0:n]
        else:
            v = np.zeros((1, n))
        svert_matrix[i] = v
    # print(svert_matrix)

    for i in range(len(such)):
        svert_matrix[len(not_such) + i][len(not_such) + i] = 1
    # print(svert_matrix)

    # неоптимизированное говно
    svert_matrix[0][len(such) + 1] = 0.23
    svert_matrix[1][len(such) + 2] = 0.1
    svert_matrix[3][len(such) + 3] = 0.21

    return svert_matrix


# получение предельной матрицы
def P_n(P_svert, n):
    s = 1
    while s > eps:
        P_svert = P_svert.dot(P_svert)
        s = 0
        for j in range(n):
            s += np.sum(P_svert[j][0:n])
    # print(' ')
    # print_matrix(P_svert)
    return P_svert


def p_vector(pi, Pn, not_such, such, locals_p):
    Pn = Pn.T
    p = Pn.dot(pi)

    n = len(not_such)

    ans = [0 for _ in not_such]
    for i in range(len(such)):
        p_vec = p[n+i] * locals_p[i]
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
        print(claster)
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


# читалка из файла
def reader(var):
    m = np.full((0, len(states)), 0)
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
    return m.astype(float)


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
    return local_p_vector


def the_forth_task(clstrs, P, locs_p):
    # свертка с поиском предельной
    P = svertka(P, clstrs[0], clstrs[1::])
    Pn = P_n(P, len(clstrs[0]))
    print_matrix(P)

    # 4.1
    for i in range(len(clstrs[0])):
        pi = [0 for _ in range(len(clstrs[0])+len(clstrs[1::]))]
        pi[i] = 1
        ans = p_vector(pi, Pn, clstrs[0], clstrs[1::],locs_p)
        print(ans)
        print(np.sum(ans))

    # 4.2
    pi = [1 / len(clusters[0]) for _ in range(len(clstrs[0]))] + [0 for _ in clusters[1::]]
    ans=p_vector(pi, Pn, clstrs[0], clstrs[1::], locs_p)
    print(ans)
    print(np.sum(ans))


if __name__ == "__main__":
    m = reader(var)

    # 3 задание
    vertex_list = []
    vertex_list += clusters[0]
    locals_p = []
    for claster in clusters[1::]:
        p = the_third_task(claster, m)
        locals_p.append(p)
        vertex_list += claster

    # сортировка по кластерам
    m, clusters = change_matrix(vertex_list, m, clusters)
    # print(clusters)

    the_forth_task(clusters, m, locals_p)

    # отрисовка переходов
    plt.grid()
    plt.ylabel("Состояния")
    plt.xlabel("Переходы")
    plt.yticks(np.arange(0, len(states) + 1, step=1))

    steps = 10
    x = [i + 1 for i in range(n)]

    for i in range(steps):
        for j in [1, 2, 3, 4, 10, 8, 15]:
            myDictionary, tr = mark_iter(n, m, states, j)
            plt.plot(x, tr, 'o--', linewidth=0.5, markeredgewidth=0.1)
    plt.show()
