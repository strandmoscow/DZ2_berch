# coding=utf8


def get_clusters(matrix):
    clusters = []

    for i in range(len(matrix)):
        if matrix[i][i] and matrix[i] not in clusters:
            clusters.append(matrix[i])

    clust_map = {}
    for cn, c in enumerate(clusters, 1):
        for x in [i for i, x in enumerate(c) if x]:
            clust_map[cn] = clust_map.get(cn, []) + [x + 1]
    return clust_map


def clusters_to_output(clusters):
    str_out = ""
    for k, v in clusters.items():
        str_out = str_out + '{}, {}\n'.format(k, v)
    return str_out
