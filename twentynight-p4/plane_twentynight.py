from decimal import Decimal, getcontext

from vector_twentynight import Vector

getcontext().prec = 30


class Plane(object):

  NO_NONZERO_ELTS_FOUND_MSG = 'No nonzero elements found'

  def __init__(self, normal_vector=None, constant_term=None):
    self.dimension = 3

    if not normal_vector:
      all_zeros = ['0']*self.dimension
      normal_vector = Vector(all_zeros)
    self.normal_vector = normal_vector

    if not constant_term:
      constant_term = Decimal('0')
    self.constant_term = Decimal(constant_term)

    self.set_basepoint()


  def set_basepoint(self):
    try:
      n = self.normal_vector.coordinates
      c = self.constant_term
      basepoint_coords = ['0']*self.dimension

      initial_index = Plane.first_nonzero_index(n)
      initial_coefficient = n[initial_index]

      basepoint_coords[initial_index] = c/initial_coefficient
      self.basepoint = Vector(basepoint_coords)

    except Exception as e:
      if str(e) == Plane.NO_NONZERO_ELTS_FOUND_MSG:
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
      initial_index = Plane.first_nonzero_index(n)
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

  
  def is_parallel_to(self, plane_2nd):
    return self.normal_vector.is_parallel_to(plane_2nd.normal_vector)

  # 两平面重合，则两平面相等
  def __eq__(self, plane_2nd):

    # 判断法向量是否为零向量
    if self.normal_vector.is_zero_vector():
      if plane_2nd.normal_vector.is_zero_vector():
        # 此时两条平面的法向量都是零向量，检查常数项是否相等，相等则为同一平面(重合)
        diff = self.constant_term - plane_2nd.constant_term
        return MyDecimal(diff).is_near_zero()
      else:
        return False
    elif plane_2nd.normal_vector.is_zero_vector():
      return False
    elif self.is_parallel_to(plane_2nd):
      p1 = self.basepoint
      p2 = plane_2nd.basepoint
      v = p2.minus(p1)
      return self.normal_vector.is_orthogonal_to(v)
    else:
      return False

  # 方程两边同时乘以常数 c
  def times_scalar(self, c):
    n_vec = self.normal_vector.times_scalar(c)
    const = self.constant_term * c
    return Plane(n_vec, const)

  # 两方程（两平面）相加
  def plus(self, p):
    n_vec = self.normal_vector.plus(p.normal_vector)
    const = self.constant_term + p.constant_term
    return Plane(n_vec, const)
  
  '''
  # 求多维交点 - 高斯消元法之前暂时注释掉
  def intersection_point_with(self, plane_2nd):
    if self.is_parallel_to(plane_2nd):
      if self == plane_2nd:
        return self
      else:
        return None
    else:
      n1 = self.normal_vector.coordinates
      n2 = plane_2nd.normal_vector.coordinates
      if MyDecimal(n1[0]).is_near_zero():
        a = n2[0]
        b = n2[1]
        c = n1[0]
        d = n1[1]
        k1 = plane_2nd.constant_term
        k2 = self.constant_term
      else:
        a = n1[0]
        b = n1[1]
        c = n2[0]
        d = n2[1]
        k1 = self.constant_term
        k2 = plane_2nd.constant_term
      
      intst_x = (d*k1 - b*k2)/(a*d - b*c)
      intst_y = (a*k2 - c*k1)/(a*d - b*c)

      return Vector([intst_x, intst_y])
    '''

  @staticmethod
  def first_nonzero_index(iterable):
    for k, item in enumerate(iterable):
      if not MyDecimal(item).is_near_zero():
          return k
    raise Exception(Plane.NO_NONZERO_ELTS_FOUND_MSG)


class MyDecimal(Decimal):
  def is_near_zero(self, eps=1e-10):
    return abs(self) < eps

def eq_para_or_notpara(p1, p2):
  if p1 == p2:
    return "Equal"
  elif p1.is_parallel_to(p2):
    return "Parallel but Unequal"
  else:
    return "Not Parallel"



'''
# test - Intersection Status
p1 = Plane(Vector([-0.412, 3.806, 0.728]), -3.46)
p2 = Plane(Vector([1.03, -9.515, -1.82]), 8.65)
print(eq_para_or_notpara(p1, p2))

p3 = Plane(Vector([2.611, 5.528, 0.283]), 4.6)
p4 = Plane(Vector([7.715, 8.306, 5.342]), 3.76)
print(eq_para_or_notpara(p3, p4))

p5 = Plane(Vector([-7.926, 8.625, -7.212]), -7.952)
p6 = Plane(Vector([-2.642, 2.875, -2.404]), -2.443)
print(eq_para_or_notpara(p5, p6))
'''