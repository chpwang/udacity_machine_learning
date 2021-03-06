from decimal import Decimal, getcontext
from copy import deepcopy

from vector_twentynight import Vector
from plane_twentynight import Plane

getcontext().prec = 30


class LinearSystem(object):

  ALL_PLANES_MUST_BE_IN_SAME_DIM_MSG = 'All planes in the system should live in the same dimension'
  NO_SOLUTIONS_MSG = 'No solutions found'
  INF_SOLUTIONS_MSG = 'Infinitely many solutions found'

  def __init__(self, planes):
    try:
      d = planes[0].dimension
      for p in planes:
          assert p.dimension == d

      self.planes = planes
      self.dimension = d

    except AssertionError:
      raise Exception(self.ALL_PLANES_MUST_BE_IN_SAME_DIM_MSG)


  def swap_rows(self, row1, row2):
    # 这个操作改变了 self 本身，不再是传值效果
    self.planes[row1], self.planes[row2] = self.planes[row2], self.planes[row1]
    return self


  def multiply_coefficient_and_row(self, coefficient, row):
    # 这个操作改变了 self 本身，不再是传值效果
    self.planes[row] = self.planes[row].times_scalar(coefficient)
    return self


  def add_multiple_times_row_to_row(self, coefficient, row_to_add, row_to_be_added_to):
    plane_to_add = self.planes[row_to_add].times_scalar(coefficient)
    # 这个操作改变了 self 本身，不再是传值效果
    self.planes[row_to_be_added_to] = self.planes[row_to_be_added_to].plus(plane_to_add)
    return self

  # 返回每一行（每一个方程）的第一个非零项的索引
  def indices_of_first_nonzero_terms_in_each_row(self):
    num_equations = len(self)
    indices = [-1] * num_equations

    for i,p in enumerate(self.planes):
      try:
        indices[i] = p.first_nonzero_index(p.normal_vector.coordinates)
      except Exception as e:
        if str(e) == Plane.NO_NONZERO_ELTS_FOUND_MSG:
            continue # 这个 continue 决定了即使找不到非零项，也不会报错
        else:
            raise e

    return indices


  # 消除第 term_index 列的所有其他项（消去除第 base_row 行之外，所有大于 i 行的方程的第 term_index 项）
  def eliminate_other_column_term_from_row_to_row(self, base_row, star_row, end_row, step, term_index):

    for j in range(star_row, end_row, step):
      c = self[j].normal_vector.coordinates[term_index]
      if not MyDecimal(c).is_near_zero():
        # 如果第 j 行，第 k 项的系数非零，则用第 i 行（i < j）消去此 j 行的第 k 项
        irow_term_onz_coef = self[base_row].normal_vector.coordinates[term_index]
        jrow_term_onz_coef = self[j].normal_vector.coordinates[term_index]
        coef_ji = jrow_term_onz_coef / irow_term_onz_coef * (-1)
        self.add_multiple_times_row_to_row(coef_ji, base_row, j)
    
    return self
  

  def compute_triangular_form(self):
    new_linearSys = deepcopy(self)
    k = 0  # 初始化, 从位置为 0 的变量（即 x1）开始消元

    for i in range(len(new_linearSys)-1):
      # 每次循环都更新矩阵里每一行第一个非零项的位置
      nonz_indices = new_linearSys.indices_of_first_nonzero_terms_in_each_row()

      while k < new_linearSys.dimension:
        # 若 i 行的第 k 项的系数为零，则寻找之后的某一行，其存在第 k 项的系数不为零，换位（swap），确保第 i 行第 k 项系数非零
        if nonz_indices[i] != k:
          if k in nonz_indices:
            row_to_swap = nonz_indices.index(i)
            new_linearSys.swap_rows(i, row_to_swap)
            # 换位后更新各行第一个非零项的 位置 列表
            nonz_indices = new_linearSys.indices_of_first_nonzero_terms_in_each_row()
            break
          else:
            # 如果所有行的第 k 项的系数都为 0 ，整个方程组无需消去第 k 项（xk），则直接跳到下一次循环，
            k += 1
            continue
        break

      if k >= new_linearSys.dimension:
        break
      # 消除第 k 列的所有其他项（消去除第 i 行之外，所有方程的第 k 项）
      new_linearSys.eliminate_other_column_term_from_row_to_row(i, i+1, len(new_linearSys), 1, k)
      k += 1
      
      

      '''
      # 我的另一种方法 - 区别参考最下方的测试代码
      # 若在 i 行之后的某一行里，存在第 i 项的系数不为零，则换位（swap），确保第 i 行第 i 项系数非零
      if nonz_indices[i] != i:
        if i in nonz_indices:
          row_to_swap = nonz_indices.index(i)
          new_linearSys.swap_rows(i, row_to_swap)
          # 换位后更新各行第一个非零项的 位置 列表
          nonz_indices = new_linearSys.indices_of_first_nonzero_terms_in_each_row()
        else:
          # 如果所有行的第 i 项的系数都为 0 ，则直接跳到下一次循环
          continue

      for j in range(i+1, len(new_linearSys)):
        if nonz_indices[j] == i:
          # 如果第 j 行，第 i 项的系数非零，则用第 i 行（i < j）消去此 j 行的第 i 项
          irow_i_term_coef = new_linearSys[i].normal_vector.coordinates[i]
          jrow_i_term_coef = new_linearSys[j].normal_vector.coordinates[i]
          coef = jrow_i_term_coef / irow_i_term_coef * (-1)
          new_linearSys.add_multiple_times_row_to_row(coef, i, j)
      '''

    return new_linearSys

  
  def compute_rref(self):
    tri_form = self.compute_triangular_form()

    for i in range(len(tri_form)-1, -1, -1):
      n_vec = tri_form[i].normal_vector
      const_term = tri_form[i].constant_term

      if n_vec.is_zero_vector():
        # 判断方程是否是 0=k 或者 0=0 的形式
        if not MyDecimal(const_term).is_near_zero():
          tri_form[i] = Plane(constant_term=1)
        else:
          continue
      else:
        first_nonz_ind = Plane.first_nonzero_index(n_vec.coordinates)
        coef = n_vec.coordinates[first_nonz_ind]
        tri_form.multiply_coefficient_and_row(1/coef, i)

        # 消除第 first_nonz_ind 列的所有其他项（消去除第 i 行之外，所有小于 i 行的方程的第 first_nonz_ind 项）
        tri_form.eliminate_other_column_term_from_row_to_row(i, i-1, -1, -1, first_nonz_ind)

    return tri_form

  
  def compute_solution(self):
    rrf_form = self.compute_rref()
    rrf_form.__raise_exception_if_no_solution()

    output = ''
    if rrf_form.__raise_exception_if_there_are_infinite_solutions():
      # raise Exception(self.INF_SOLUTIONS_MSG)
      para = rrf_form.parameterize_for_infinite_solution()
      output += "Infinite solutions found...\n{}".format(para)
    else:
      for i in range(self.dimension):
        output += "x{} = {} \n".format(i+1, round(rrf_form[i].constant_term, 3))
    
    return output

  
  def __raise_exception_if_no_solution(self):
    # 若法向量为零向量，且常数项不为零，则抛出异常提示无解
    for p in self.planes:
      try:
        p.first_nonzero_index(p.normal_vector.coordinates)
      except Exception as e:
        # 查看报错信息是否和 Plane 类型里的报错一样
        if str(e) == 'No nonzero elements found':
          const_t = MyDecimal(p.constant_term)
          if not const_t.is_near_zero(): 
            raise Exception(self.NO_SOLUTIONS_MSG)
        else:
          raise e


  def __raise_exception_if_there_are_infinite_solutions(self):
    # 在确定有解的前提下，若第 i 行的第一个含非零系数的项不为 i ，则方程化简过程中有变量缺失，故含有多解
    first_nonz_indices = self.indices_of_first_nonzero_terms_in_each_row()
    for i in range(self.dimension):
      if i >= len(first_nonz_indices) or first_nonz_indices[i] != i:
        return True
    return False


  def parameterize_for_infinite_solution(self):
    first_nonz_indices = self.indices_of_first_nonzero_terms_in_each_row()
    pivots_num = sum([1 if ind >= 0 else 0 for ind in first_nonz_indices])
    free_var_num = self.dimension-pivots_num

    # 初始化基准点（Base Point）和各方向向量
    bpoint_list = [0]*self.dimension

    dir_vecs = [[0]*self.dimension for k in range(free_var_num)]

    
    for row, i in enumerate(first_nonz_indices):
      if i >= 0:
        bpoint_list[i] = self[row].constant_term

        free_var_ind = 0
        for j in range(i+1, self.dimension):
          coef = self[row].normal_vector.coordinates[j]
          if not MyDecimal(coef).is_near_zero():
            dir_vecs[free_var_ind][i] = coef * (-1)
            dir_vecs[free_var_ind][j] = 1
            free_var_ind += 1

    for ind, dv in enumerate(dir_vecs):
      dir_vecs[ind] = Vector(dv)
    bpoint = Vector(bpoint_list)

    return Parameterization(bpoint, dir_vecs)
      
        

  def __len__(self):
    return len(self.planes)

  # __getitem__ 这个函数使得对于 s = LinearSystem([p0,p1,p2]) ，可使用索引方式 s[1] 得到 p1
  def __getitem__(self, i):
    return self.planes[i]

  # __setitem__ 这个函数使得对于 s = LinearSystem([p0,p1,p2]) ，可使用索引方式 s[1] = p3 将 s[1] 的值设置为 p3
  def __setitem__(self, i, x):
    try:
      assert x.dimension == self.dimension
      self.planes[i] = x

    except AssertionError:
      raise Exception(self.ALL_PLANES_MUST_BE_IN_SAME_DIM_MSG)


  def __str__(self):
    ret = 'Linear System:\n'
    temp = ['Equation {}: {}'.format(i+1,p) for i,p in enumerate(self.planes)]
    ret += '\n'.join(temp)
    return ret

# 此类（Class）用来判断某数是否为零
class MyDecimal(Decimal):
  def is_near_zero(self, eps=1e-10):
    return abs(self) < eps


class Parameterization(object):

  def __init__(self, basepoint, direction_vectors):

    self.base_point = basepoint
    self.direction_vectors = direction_vectors
    self.dimension = basepoint.dimension

  def __str__(self):
    ret = "Parametric Equations:"
    temp = ""
    for i in range(self.dimension):
      temp += "\nx{} = {}".format(i+1, round(self.base_point.coordinates[i], 3))
      
      for j, v in enumerate(self.direction_vectors):
        c = v.coordinates[i]
        if MyDecimal(c).is_near_zero():
          continue
        elif c < 0:
          op = "-"
        else:
          op = "+"
        temp += " {} {}*t{}".format(op, round(abs(c), 3), j+1)

    ret += temp
    return ret
    



# example - 解线性方程组（Linear System）
# Linear System:
# Equation 1: -7x_1 + 5x_2 - x_3 = 1
# Equation 2: x_1 - 3x_2 - 8x_3 = 1
# Equation 3: -10x_1 - 2x_2 + 9x_3 = 1

p1 = Plane(normal_vector=Vector([-7,5,-1]), constant_term=1)
p2 = Plane(normal_vector=Vector([1,-3,-8]), constant_term=1)
p3 = Plane(normal_vector=Vector([-10,-2,9]), constant_term=1)
s = LinearSystem([p1,p2,p3])
print(s)
print(s.compute_rref())
print(s.compute_solution())



'''
# test - parameterization output for infinite solutions
p1 = Plane(normal_vector=Vector([0.786, 0.786, 0.588]), constant_term=-0.714)
p2 = Plane(normal_vector=Vector([-0.131, -0.131, 0.244]), constant_term=0.319)
s = LinearSystem([p1,p2])
print(s.compute_rref())
print(s.compute_solution())


p1 = Plane(normal_vector=Vector([8.631, 5.112, -1.816]), constant_term=-5.113)
p2 = Plane(normal_vector=Vector([4.315, 11.132, -5.27]), constant_term=-6.775)
p3 = Plane(normal_vector=Vector([-2.158, 3.01, -1.727]), constant_term=-0.831)
s = LinearSystem([p1,p2,p3])
print(s.compute_rref())
print(s.compute_solution())


p1 = Plane(normal_vector=Vector([0.935, 1.76, -9.365]), constant_term=-9.955)
p2 = Plane(normal_vector=Vector([0.187, 0.352, -1.873]), constant_term=-1.991)
p3 = Plane(normal_vector=Vector([0.374, 0.704, -3.746]), constant_term=-3.982)
p4 = Plane(normal_vector=Vector([-0.561, -1.056, 5.619]), constant_term=5.973)
s = LinearSystem([p1,p2,p3,p4])
print(s.compute_rref())
print(s.compute_solution())
'''



'''
# test - function compute_solution()
p1 = Plane(normal_vector=Vector([5.862, 1.178, -10.366]), constant_term=-8.15)
p2 = Plane(normal_vector=Vector([-2.931, -0.589, 5.183]), constant_term=-4.075)
s = LinearSystem([p1,p2])
print(s.compute_solution())


p1 = Plane(normal_vector=Vector([8.631, 5.112, -1.816]), constant_term=-5.113)
p2 = Plane(normal_vector=Vector([4.315, 11.132, -5.27]), constant_term=-6.775)
p3 = Plane(normal_vector=Vector([-2.158, 3.01, -1.727]), constant_term=-0.831)
s = LinearSystem([p1,p2,p3])
print(s.compute_solution())


p1 = Plane(normal_vector=Vector([5.262, 2.739, -9.878]), constant_term=-3.441)
p2 = Plane(normal_vector=Vector([5.111, 6.358, 7.638]), constant_term=-2.152)
p3 = Plane(normal_vector=Vector([2.016, -9.924, -1.367]), constant_term=-9.278)
p4 = Plane(normal_vector=Vector([2.167, -13.543, -18.883]), constant_term=-10.567)
s = LinearSystem([p1,p2,p3,p4])
print(s.compute_solution())
'''



'''
# test - function compute_rref()
p1 = Plane(normal_vector=Vector(['1','1','1']), constant_term='1')
p2 = Plane(normal_vector=Vector(['0','1','1']), constant_term='2')
s = LinearSystem([p1,p2])
print(s)
r = s.compute_rref()
print(r)
if not (r[0] == Plane(normal_vector=Vector(['1','0','0']), constant_term='-1') and
        r[1] == p2):
    print('test case 1 failed')
else:
  print('test case 1 success')


p1 = Plane(normal_vector=Vector(['1','1','1']), constant_term='1')
p2 = Plane(normal_vector=Vector(['1','1','1']), constant_term='2')
s = LinearSystem([p1,p2])
print(s)
r = s.compute_rref()
print(r)
if not (r[0] == p1 and
        r[1] == Plane(constant_term='1')):
    print('test case 2 failed')
else:
  print('test case 2 success')

p1 = Plane(normal_vector=Vector(['1','1','1']), constant_term='1')
p2 = Plane(normal_vector=Vector(['0','1','0']), constant_term='2')
p3 = Plane(normal_vector=Vector(['1','1','-1']), constant_term='3')
p4 = Plane(normal_vector=Vector(['1','0','-2']), constant_term='2')
s = LinearSystem([p1,p2,p3,p4])
print(s)
r = s.compute_rref()
print(r)
if not (r[0] == Plane(normal_vector=Vector(['1','0','0']), constant_term='0') and
        r[1] == p2 and
        r[2] == Plane(normal_vector=Vector(['0','0','-2']), constant_term='2') and
        r[3] == Plane()):
    print('test case 3 failed')
else:
  print('test case 3 success')

p1 = Plane(normal_vector=Vector(['0','1','1']), constant_term='1')
p2 = Plane(normal_vector=Vector(['1','-1','1']), constant_term='2')
p3 = Plane(normal_vector=Vector(['1','2','-5']), constant_term='3')
s = LinearSystem([p1,p2,p3])
print(s)
r = s.compute_rref()
print(r)
if not (r[0] == Plane(normal_vector=Vector(['1','0','0']), constant_term=Decimal('23')/Decimal('9')) and
        r[1] == Plane(normal_vector=Vector(['0','1','0']), constant_term=Decimal('7')/Decimal('9')) and
        r[2] == Plane(normal_vector=Vector(['0','0','1']), constant_term=Decimal('2')/Decimal('9'))):
    print('test case 4 failed')
else:
  print('test case 4 success')
'''







'''
# test - 针对测试我写的另一种 compute_triangular_form() 方法
# 可以运行查看和 Udacity 老师写法区别
p1 = Plane(normal_vector=Vector(['1','0','1']), constant_term='1')
p2 = Plane(normal_vector=Vector(['0','0','1']), constant_term='2')
p3 = Plane(normal_vector=Vector(['1','0','-1']), constant_term='3')
p4 = Plane(normal_vector=Vector(['1','0','-2']), constant_term='2')
p5 = Plane(normal_vector=Vector(['0','0','0']), constant_term='2')
s = LinearSystem([p1,p2,p3,p4,p5])
print(s)
t = s.compute_triangular_form()
print(t)
'''

'''
# test - function compute_triangular_form()
p1 = Plane(normal_vector=Vector(['1','1','1']), constant_term='1')
p2 = Plane(normal_vector=Vector(['0','1','1']), constant_term='2')
s = LinearSystem([p1,p2])
print(s)
t = s.compute_triangular_form()
print(t)
if not (t[0] == p1 and
        t[1] == p2):
    print('test case 1 failed')
else:
  print('test case 1 success')

p1 = Plane(normal_vector=Vector(['1','1','1']), constant_term='1')
p2 = Plane(normal_vector=Vector(['1','1','1']), constant_term='2')
s = LinearSystem([p1,p2])
print(s)
t = s.compute_triangular_form()
print(t)
if not (t[0] == p1 and
        t[1] == Plane(constant_term='1')):
    print('test case 2 failed')
else:
  print('test case 2 success')

p1 = Plane(normal_vector=Vector(['1','1','1']), constant_term='1')
p2 = Plane(normal_vector=Vector(['0','1','0']), constant_term='2')
p3 = Plane(normal_vector=Vector(['1','1','-1']), constant_term='3')
p4 = Plane(normal_vector=Vector(['1','0','-2']), constant_term='2')
s = LinearSystem([p1,p2,p3,p4])
print(s)
t = s.compute_triangular_form()
print(t)
if not (t[0] == p1 and
        t[1] == p2 and
        t[2] == Plane(normal_vector=Vector(['0','0','-2']), constant_term='2') and
        t[3] == Plane()):
    print('test case 3 failed')
else:
  print('test case 3 success')

p1 = Plane(normal_vector=Vector(['0','1','1']), constant_term='1')
p2 = Plane(normal_vector=Vector(['1','-1','1']), constant_term='2')
p3 = Plane(normal_vector=Vector(['1','2','-5']), constant_term='3')
s = LinearSystem([p1,p2,p3])
print(s)
t = s.compute_triangular_form()
print(t)
if not (t[0] == Plane(normal_vector=Vector(['1','-1','1']), constant_term='2') and
        t[1] == Plane(normal_vector=Vector(['0','1','1']), constant_term='1') and
        t[2] == Plane(normal_vector=Vector(['0','0','-9']), constant_term='-2')):
    print('test case 4 failed')
else:
  print('test case 4 success')
'''





'''
# test - function swap_rows(), multiply_coefficient_and_row() and add_multiple_times_row_to_row()
p0 = Plane(normal_vector=Vector(['1','1','1']), constant_term='1')
p1 = Plane(normal_vector=Vector(['0','1','0']), constant_term='2')
p2 = Plane(normal_vector=Vector(['1','1','-1']), constant_term='3')
p3 = Plane(normal_vector=Vector(['1','0','-2']), constant_term='2')

s = LinearSystem([p0,p1,p2,p3])

s.swap_rows(0,1)
if not (s[0] == p1 and s[1] == p0 and s[2] == p2 and s[3] == p3):
    print('test case 1 failed')
else:
  print('test case 1 success')

s.swap_rows(1,3)
if not (s[0] == p1 and s[1] == p3 and s[2] == p2 and s[3] == p0):
    print('test case 2 failed')
else:
  print('test case 2 success')

s.swap_rows(3,1)
if not (s[0] == p1 and s[1] == p0 and s[2] == p2 and s[3] == p3):
    print('test case 3 failed')
else:
  print('test case 3 success')    


s.multiply_coefficient_and_row(1,0)
if not (s[0] == p1 and s[1] == p0 and s[2] == p2 and s[3] == p3):
    print('test case 4 failed')
else:
  print('test case 4 success') 


s.multiply_coefficient_and_row(-1,2)
if not (s[0] == p1 and
        s[1] == p0 and
        s[2] == Plane(normal_vector=Vector(['-1','-1','1']), constant_term='-3') and
        s[3] == p3):
    print('test case 5 failed')
else:
   print("test case 5 success")


s.multiply_coefficient_and_row(10,1)
if not (s[0] == p1 and
        s[1] == Plane(normal_vector=Vector(['10','10','10']), constant_term='10') and
        s[2] == Plane(normal_vector=Vector(['-1','-1','1']), constant_term='-3') and
        s[3] == p3):
    print('test case 6 failed')
else:
   print("test case 6 success")


s.add_multiple_times_row_to_row(0,0,1)
if not (s[0] == p1 and
        s[1] == Plane(normal_vector=Vector(['10','10','10']), constant_term='10') and
        s[2] == Plane(normal_vector=Vector(['-1','-1','1']), constant_term='-3') and
        s[3] == p3):
    print('test case 7 failed')
else:
   print("test case 7 success")



s.add_multiple_times_row_to_row(1,0,1)
if not (s[0] == p1 and
        s[1] == Plane(normal_vector=Vector(['10','11','10']), constant_term='12') and
        s[2] == Plane(normal_vector=Vector(['-1','-1','1']), constant_term='-3') and
        s[3] == p3):
    print('test case 8 failed')
else:
   print("test case 8 success")

s.add_multiple_times_row_to_row(-1,1,0)
if not (s[0] == Plane(normal_vector=Vector(['-10','-10','-10']), constant_term='-10') and
        s[1] == Plane(normal_vector=Vector(['10','11','10']), constant_term='12') and
        s[2] == Plane(normal_vector=Vector(['-1','-1','1']), constant_term='-3') and
        s[3] == p3):
    print('test case 9 failed')
else:
   print("test case 9 success")
'''




'''
# test output - 测试输出
p0 = Plane(normal_vector=Vector(['1','1','1']), constant_term='1')
p1 = Plane(normal_vector=Vector(['0','1','0']), constant_term='2')
p2 = Plane(normal_vector=Vector(['1','1','-1']), constant_term='3')
p3 = Plane(normal_vector=Vector(['1','0','-2']), constant_term='2')

s = LinearSystem([p0,p1,p2,p3])

print(s.indices_of_first_nonzero_terms_in_each_row())
print('{},{},{},{}'.format(s[0],s[1],s[2],s[3]))
print(len(s))
print(s)

s[0] = p1
print(s)

print(MyDecimal('1e-9').is_near_zero())
print(MyDecimal('1e-11').is_near_zero())
'''