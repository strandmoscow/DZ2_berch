import numpy as np
from jinja2 import Environment, FileSystemLoader
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM


def matr_to_table(mat):
    s = ""
    # for i in range(len(mat)):
    #     for j in range(len(mat[0])):
    #         s = s + f"{str(mat[i][j])}"
    #         if j < len(mat[0])-1:
    #             s = s + " & "
    #     if i < len(mat)-1:
    #         s = s + " \\\\\n"
    for i in mat:
        for x in i:
           s += f'{str(x)}'
           s += ' & '
        s = s[:-3]
        s = s + " \\\\\n"
    s = s[:-3]
    return s


def matr_to_table_r(mat, n, delim):
    s = ""
    # for i in range(len(mat)):
    #     for j in range(len(mat[0])):
    #         s = s + f"{str(mat[i][j])}"
    #         if j < len(mat[0])-1:
    #             s = s + " & "
    #     if i < len(mat)-1:
    #         s = s + " \\\\\n"
    for i in mat:
        for x in i:
           s += f'{round(x, n)}'
           s += f'{delim}'
        s = s[:-len(delim)]
        s = s + " \\\\\n"
    s = s[:-3]
    return s


def matr_to_table_r_gr(mat, n, delim):
    s = ""
    for i in mat:
        for x in i:
           s += f'{round(x, n)}'
           s += f'{delim}'
        s = s[:-len(delim)]
        s = s + " \n"
    s = s[:-2]
    return s


def tabl_to_table(mat, n):
    s = ""
    for i in range(len(mat) - 1):
        s = s + f"{i+1} & {mat[i][1] / n} & {mat[i][2] / n} & {mat[i][3] / n} & {mat[i][4] / n} & {mat[i][5] / n} \\\\\n \\hline "
    s = s + f"{len(mat)} & {mat[len(mat) - 1][1] / n} & {mat[len(mat) - 1][2] / n} & {mat[len(mat) - 1][3] / n} & {mat[len(mat) - 1][4] / n} & {mat[len(mat) - 1][5] / n} \\\\\n \\hline"
    return s


def vector_to_tex(v, delim):
    s = ""
    for x in v:
        s += f"{str(round(x, 4))}"
        s += f"{delim}"
    s = s[:-len(delim)]
    s = s + " \\\\\n"
    return s


def ispr_ocen(matr):
    a1 = np.empty(len(matr))
    a2 = np.empty(len(matr))
    a3 = np.empty(len(matr))
    a4 = np.empty(len(matr))
    a5 = np.empty(len(matr))
    for i in range(len(matr)):
        a1[i] = matr[i][1] / 100.
        a2[i] = matr[i][2] / 100.
        a3[i] = matr[i][3] / 100.
        a4[i] = matr[i][4] / 100.
        a5[i] = matr[i][5] / 100.

    return f"{'%.3f' % np.sqrt(20 * np.var(a1)/(20 - 1))} & {'%.3f' % np.sqrt(20 * np.var(a2)/(20 - 1))} & {'%.3f' % np.sqrt(20 * np.var(a3)/(20 - 1))} & {'%.3f' % np.sqrt(20 * np.var(a4) / (20 - 1))} & {'%.3f' % np.sqrt(20 * np.var(a5) / (20 - 1))}"


def lols_tex(lols):
    s = ""
    for lol in lols:
        s = s + f"{ round(lol / 150, 3) } ,"
    return s[:-2]


def make_latex(var_num, group, fullname, fullname_short, stohast_matr, rgo, locals_p, m_svert, P, P_n, ans1, ans2, list_of_last_states):
    # Jinja init
    environment = Environment(
        loader=FileSystemLoader("Latex/templates/")
    )

    # Preamble text
    base_template = environment.get_template("educmm_lab_Variant_N_M-id.tex")
    base_res_file_name = "Latex/res/labs/educmm_txb_COMPMATHLAB-Solution_N_M/educmm_lab_Variant_N_M-id.tex"
    base_text = base_template.render(
        author_name=fullname,
        author_name_short=fullname_short,
        group=group,
        variant=var_num
    )

    with open(base_res_file_name, mode="w+", encoding="utf-8") as base:
        base.write(base_text)
        print(f"... wrote {base_res_file_name}")

    # Main text
    latex_text_template = environment.get_template("educmm_txb_COMPMATHLAB-Solution_N_M.tex")
    latex_text_file_name = f"Latex/res/labs/educmm_txb_COMPMATHLAB-Solution_N_M.tex"
    latex_text = latex_text_template.render(
        stohast_matr=matr_to_table_r(stohast_matr, 2, " & "),
        r=matr_to_table(rgo[0]),
        r_vec=vector_to_tex(locals_p[0], "\quad"),
        g=matr_to_table(rgo[1]),
        g_vec=vector_to_tex(locals_p[1], "\quad"),
        o=matr_to_table(rgo[2]),
        o_vec=vector_to_tex(locals_p[2], "\quad"),
        stohast_matr_svert=matr_to_table_r(m_svert, 3, " & "),
        P=matr_to_table_r(P, 3, " & "),
        P_n=matr_to_table_r(P_n, 3, " & "),
        ans1_1=vector_to_tex(ans1[0], ', '),
        ans1_2=vector_to_tex(ans1[1], ', '),
        ans1_3=vector_to_tex(ans1[2], ', '),
        ans1_4=vector_to_tex(ans1[3], ', '),
        ans2=vector_to_tex(ans2, ', '),
        lols_tex=lols_tex(list_of_last_states)
    )
    with open(latex_text_file_name, mode="w+", encoding="utf-8") as text:
        text.write(latex_text)
        print(f"... wrote {latex_text_file_name}")

    # print("\nКласическая матрица:")
    # print(matr_to_table_r_gr(stohast_matr, 3, ", "))
    # print("\n")

    print("\nМодифицированная матрица:")
    print(matr_to_table_r_gr(m_svert, 3, ", "))
    print("\n")

    print("\nСвернутая матрица:")
    print(matr_to_table_r_gr(P, 3, ", "))
    print("\n")


