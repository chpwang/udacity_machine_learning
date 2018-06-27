
from fractions import Fraction
from copy import deepcopy

I = [[9,13,5,2], 
     [1,11,7,6], 
     [3,7,4,1],
     [6,0,7,10]]

A = [[1,2,3], 
     [2,3,3], 
     [1,2,5]]

B = [[1,2,3,5], 
     [2,3,3,5], 
     [1,2,5,1]]

C = [[1],
     [2],
     [3]]


def matxRound(M, decPts=0):
  for i in range(len(M)):
    for j,e in enumerate(M[i]):
      M[i][j] = round(e, decPts)

def transpose(M):
  t_M = list(zip(*M))
  for i in range(len(t_M)):
    t_M[i] = list(t_M[i])
  return t_M

def shape(M):
    rows_num = len(M)
    if rows_num == 0:
        columns_num = 0
    else:
        columns_num = len(M[0])
    return (rows_num, columns_num)


def matxMultiply(A, B):
    a_shp = shape(A)
    t_B = transpose(B)
    t_b_shp = shape(t_B)
    new_matx = []

    # 1st columns = 2nd rows
    if a_shp[1] != t_b_shp[1]:
      raise ValueError("Can't mutiply the two Matrixes, please check the input!")
    else:
      try:

        for i in range(a_shp[0]):
          row = []
          for j in range(t_b_shp[0]):
            row.append(sum([x*y for x,y in zip(A[i], t_B[j])]))

          new_matx.append(row)
      except Exception as e:
        raise e
      else:
        return new_matx


def augmentMatrix(A, b):
    a_shp = shape(A)
    b_shp = shape(b)
    if a_shp[0] != b_shp[0] or b_shp[1] != 1:
        raise ValueError("column and row doesn't match. check the inputs.")
    aug_matx = []
    for i in range(a_shp[0]):
        aug_matx.append(A[i] + b[i])

    return aug_matx

def swapRows(M, r1, r2):
    temp = M[r1]
    M[r1] = M[r2]
    M[r2] = temp

def scaleRow(M, r, scale):
    if abs(scale) < 1.0e-9:
      raise ValueError("Can't scale a row to zero, check the input")
    M[r] = [x*scale for x in M[r]]

def addScaledRow(M, r1, r2, scale=1):
    M[r1] = [x+y*scale for x,y in zip(M[r1], M[r2])]


def find_max_abs_beneath_and_include_row(row, matrx):
    temp_m = list(zip(*matrx))
    ben_n_inc_li = temp_m[row][row:]
    abs_ben_n_inc_li = [abs(x) for x in ben_n_inc_li]
    max_row = abs_ben_n_inc_li.index(max(abs_ben_n_inc_li))+row
    return max_row

def eliminate_other_element_in_column(col, matrx, tolerance=1.0e-9):
    temp_m = list(zip(*matrx))
    col_to_be_elim = temp_m[col]
    scales_li = [x*(-1)/col_to_be_elim[col] for x in col_to_be_elim]
    for i, scl in enumerate(scales_li):
        if abs(scl) < tolerance or i == col:
            continue
        else:
            addScaledRow(matrx, i, col, scl)
        
def gj_Solve(A, b, decPts=4, epsilon=1.0e-16):
    a_shp = shape(A)
    b_shp = shape(b)
    if a_shp[0] != b_shp[0] or a_shp == (0, 0) or a_shp[0] < a_shp[1] or b_shp[1] != 1:
        return None
    aug_matrx = augmentMatrix(A, b)
    matxRound(aug_matrx, decPts)

    for j in range(a_shp[0]):
        max_elem_row = find_max_abs_beneath_and_include_row(j, aug_matrx)
        leading_coef = aug_matrx[max_elem_row][j]
        if abs(leading_coef) < epsilon:
            return None

        if j != max_elem_row:
            swapRows(aug_matrx, j, max_elem_row)

        scaleRow(aug_matrx, j, 1/leading_coef)
        eliminate_other_element_in_column(j, aug_matrx)

    solutions_to_be_format = list(list(zip(*aug_matrx))[-1])
    solutions_li = [[x] for x in solutions_to_be_format]
    return solutions_li

# 二维空间线性回归方程的最优系数 m 和 b（m 是斜率，b 是截距）- 根据 Normal Equation 原理求解
def linearRegression2D(X,Y):
    def format_matrx(matrx):
        if isinstance(matrx[0], list):
            return matrx
        else:
            return [[k] for k in matrx]

    # 格式化输入
    formated_X = format_matrx(X)
    formated_Y = format_matrx(Y)
    
    # 求解最优系数
    unit_matrx = [[1]]*len(formated_X)
    aug_X = augmentMatrix(formated_X, unit_matrx)
    linSys = matxMultiply(transpose(aug_X), aug_X)
    b = matxMultiply(transpose(aug_X), formated_Y)
    solutions_li = gj_Solve(linSys, b)

    opt_m = solutions_li[0][0]
    opt_b = solutions_li[1][0]
    return opt_m, opt_b


# 多元线性回归系数 - 根据 Normal Equation 原理求解
def linearRegression(X,Y,dimension=3):
    def format_matrx(matrx):
        if isinstance(matrx[0], list):
            return matrx
        else:
            return [[k] for k in matrx]

    def mutated_augMat(A, b):
        a_shp = shape(A)
        b_shp = shape(b)
        if a_shp[0] != b_shp[0] or b_shp[1] != 1:
            raise ValueError("column and row doesn't match. check the inputs.")
        aug_matx = []
        for i in range(a_shp[0]):
            aug_matx.append(b[i] + A[i])

        return aug_matx

    # 格式化输入
    formated_X = format_matrx(X)
    formated_Y = format_matrx(Y)
    
    # 求解最优系数
    unit_matrx = [[1]]*len(formated_X)
    aug_X = mutated_augMat(formated_X, unit_matrx)
    linSys = matxMultiply(transpose(aug_X), aug_X)
    b = matxMultiply(transpose(aug_X), formated_Y)
    solutions_li = gj_Solve(linSys, b)
    
    coeff = [k[0] for k in solutions_li]
    return coeff




''' 
# 解线性方程组（Linear System）
# ------------
# example 01
# Linear System:
# Equation 1: -7x_1 + 5x_2 - x_3 = 1
# Equation 2: x_1 - 3x_2 - 8x_3 = 1
# Equation 3: -10x_1 - 2x_2 + 9x_3 = 1

D = [[-7,5,-1], 
     [1,-3,-8], 
     [-10,-2,9]]

# 注意这里的格式，输入是 2 维 list，不是 [1,1,1]
b = [[1],
     [1],
     [1]]

print(augmentMatrix(D, b)) # 输出增广矩阵（Augmented Matrix）
print(gj_Solve(D, b)) # 输出方程的解

# ------------
# example 02 - 支持分数系数
# Linear System:
# Equation 1: 16x_1 + 6x_2= 11
# Equation 2: 6x_1 + 3x_2 = 5

E = [[Fraction(16, 1),Fraction(6, 1)], 
     [Fraction(6, 1),Fraction(3, 1)]]
e = [[Fraction(11, 1)],
     [Fraction(5, 1)]]

print(augmentMatrix(E, e)) # 输出增广矩阵（Augmented Matrix）
print(gj_Solve(E, e)) # 输出方程的解
'''


