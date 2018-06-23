from decimal import Decimal, getcontext
from copy import deepcopy

from vector_twentynight import Vector
from plane_twentynight import Plane

getcontext().prec = 30


class LinearSystem(object):

  ALL_PLANES_MUST_BE_IN_SAME_DIM_MSG = 'All planes in the system should live in the same dimension'
  NO_SOLUTIONS_MSG = 'No solutions'
  INF_SOLUTIONS_MSG = 'Infinitely many solutions'

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

  # 返回每一列（每一个方程）的第一个非零项的索引
  def indices_of_first_nonzero_terms_in_each_row(self):
    num_equations = len(self)
    # num_variables = self.dimension  # 得出有多少个未知变量需要计算

    indices = [-1] * num_equations

    for i,p in enumerate(self.planes):
      try:
        indices[i] = p.first_nonzero_index(p.normal_vector.coordinates)
      except Exception as e:
        if str(e) == Plane.NO_NONZERO_ELTS_FOUND_MSG:
            continue
        else:
            raise e

    return indices


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


class MyDecimal(Decimal):
  def is_near_zero(self, eps=1e-10):
    return abs(self) < eps












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