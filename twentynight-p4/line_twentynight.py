from decimal import Decimal, getcontext
from vector_twentynight import Vector

# 设置 Decimal 数据类型小数点后保留的位数
getcontext().prec = 30

class Line(object):

  NO_NONZERO_ELTS_FOUND_MSG = 'No nonzero elements found'
  NO_UNIQUE_INTERSECTION_POINT_FOUND_MSG = 'Two lines are parallel. No unique intersection point found'
  INFINITE_INTERSECTION_POINT_FOUND_MSG = 'Two lines are coincide. Infinite intersection points found'

  def __init__(self, normal_vector=None, constant_term=None):
    self.dimension = 2

    if not normal_vector:
      all_zeros = ['0']*self.dimension
      normal_vector = Vector(all_zeros)
    self.normal_vector = normal_vector

    if not constant_term:
      constant_term = Decimal('0')
    self.constant_term = Decimal(constant_term)

    self.set_basepoint()

  # 设置直线上的基准点（base point）
  def set_basepoint(self):
    try:
      n = self.normal_vector.coordinates
      c = self.constant_term
      basepoint_coords = ['0']*self.dimension

      initial_index = Line.first_nonzero_index(n)
      initial_coefficient = n[initial_index]

      basepoint_coords[initial_index] = c/initial_coefficient
      self.basepoint = Vector(basepoint_coords)

    except Exception as e:
      if str(e) == Line.NO_NONZERO_ELTS_FOUND_MSG:
        self.basepoint = None
      else:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
        raise e


  def __str__(self):

    num_decimal_places = 3

    def write_coefficient(coefficient, is_initial_term=False):
      coefficient = round(coefficient, num_decimal_places)
      if coefficient % 1 == 0:
          coefficient = int(coefficient)

      output = ''

      if coefficient < 0:
          output += '-'
      if coefficient > 0 and not is_initial_term:
          output += '+'

      if not is_initial_term:
          output += ' '

      if abs(coefficient) != 1:
          output += '{}'.format(abs(coefficient))

      return output

    n = self.normal_vector.coordinates

    try:
      initial_index = Line.first_nonzero_index(n)
      terms = [write_coefficient(n[i], is_initial_term=(i==initial_index)) + 'x_{}'.format(i+1)
               for i in range(self.dimension) if round(n[i], num_decimal_places) != 0]
      output = ' '.join(terms)

    except Exception as e:
      if str(e) == self.NO_NONZERO_ELTS_FOUND_MSG:
        output = '0'
      else:
        raise e

    constant = round(self.constant_term, num_decimal_places)
    if constant % 1 == 0:
      constant = int(constant)
    output += ' = {}'.format(constant)

    return output

  
  def is_parallel_to(self, line_2nd):
    return self.normal_vector.is_parallel_to(line_2nd.normal_vector)
  
  def is_coincide_with(self, line_2nd):
    if self.is_parallel_to(line_2nd):
      p1 = self.basepoint
      p2 = line_2nd.basepoint
      v = p2.minus(p1)
      return p1 == p2 or self.normal_vector.is_orthogonal_to(v)
    else:
      return False


  def intersection_point_with(self, line_2nd):
    if self.is_parallel_to(line_2nd):
      if self.is_coincide_with(line_2nd):
        return self.INFINITE_INTERSECTION_POINT_FOUND_MSG
      else:
        return self.NO_UNIQUE_INTERSECTION_POINT_FOUND_MSG
    else:
      n1 = self.normal_vector.coordinates
      n2 = line_2nd.normal_vector.coordinates
      if MyDecimal(n1[0]).is_near_zero():
        a = n2[0]
        b = n2[1]
        c = n1[0]
        d = n1[1]
        k1 = line_2nd.constant_term
        k2 = self.constant_term
      else:
        a = n1[0]
        b = n1[1]
        c = n2[0]
        d = n2[1]
        k1 = self.constant_term
        k2 = line_2nd.constant_term
      
      intst_x = (d*k1 - b*k2)/(a*d - b*c)
      intst_y = (a*k2 - c*k1)/(a*d - b*c)

      return Vector([intst_x, intst_y])


  @staticmethod
  def first_nonzero_index(iterable):
    for k, item in enumerate(iterable):
      if not MyDecimal(item).is_near_zero():
        return k
    raise Exception(Line.NO_NONZERO_ELTS_FOUND_MSG)


class MyDecimal(Decimal):
  def is_near_zero(self, eps=1e-10):
    return abs(self) < eps






'''
# test - Find Intersection Points
l1 = Line(Vector([4.046, 2.836]), 1.21)
l2 = Line(Vector([10.115, 7.09]), 3.025)
print(l1.intersection_point_with(l2))

l3 = Line(Vector([7.204, 3.182]), 8.68)
l4 = Line(Vector([8.172, 4.114]), 9.883)
print(l3.intersection_point_with(l4))

l5 = Line(Vector([1.182, 5.562]), 6.744)
l6 = Line(Vector([1.773, 8.343]), 9.525)
print(l5.intersection_point_with(l6))
'''