# coding=utf8

import sys

from public.mcl_manager import mcl_manager
from public.matrix import MatrixException


def clusterisation(file_name):

    try:
        filename = file_name
    except:
        raise Exception('Забыли ввести имя файла на вход!')

    output = sys.argv[2] if len(sys.argv) > 2 else 'output'

    return mcl_manager(filename)
