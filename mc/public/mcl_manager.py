# coding=utf8

import time

from mc.public.mcl import mcl
from mc.public.clusters import clusters_to_output


def reader(filename):
    matrix = []
    with open(filename) as input_file:
        for r in input_file.readlines():
            values = r.strip().split(",")
            matrix.append([float(x.strip()) for x in values])
    return matrix


def mcl_manager(input_file):
    matrix = reader(input_file)

    begin_time = time.time()
    matrix, clusters, loops_count = mcl(matrix)
    end_time = time.time()

    print("Вершин в графе: {}".format(len(matrix[0])))
    print("Посчитано за {} секунд".format(end_time - begin_time))
    print("Программа сработала за {} циклов\n".format(loops_count))
    return clusters
