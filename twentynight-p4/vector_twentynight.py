import math
from decimal import Decimal, getcontext

# 设置 Decimal 数据类型小数点后保留的位数
getcontext().prec = 19
# 设置公差，用于判断变量是否为零
TOLERANCE = 1e-10

class Vector(object):
  # 实例（instance）的创建
  def __init__(self, coordinates):
    try:
      if not coordinates:
          raise ValueError
      self.coordinates = tuple([Decimal(x) for x in coordinates])
      self.dimension = len(coordinates)

    except ValueError:
      raise ValueError('The coordinates must be nonempty')

    except TypeError:
      raise TypeError('The coordinates must be an iterable')

  # print 输出内容
  def __str__(self):
    return 'Vector: {}'.format(self.coordinates)

  # 定义相等
  def __eq__(self, v):
    return self.coordinates == v.coordinates

  def plus(self, v):
    new_coordinates = [x+y for x,y in zip(self.coordinates, v.coordinates)]
    return Vector(new_coordinates)

  def minus(self, v):
    return self.plus(v.times_scalar(-1))

  def times_scalar(self, c):
    new_coordinates = [x*c for x in self.coordinates]
    return Vector(new_coordinates)

  def dot_product_with(self, v):
    new_coordinates = [x*y for x,y in zip(self.coordinates, v.coordinates)]
    return sum(new_coordinates)

  def magnitude(self):
    return Decimal(math.sqrt(self.dot_product_with(self)))

  def normalization(self):
    try:
      mag = self.magnitude()
      return self.times_scalar(Decimal(1)/mag)
    except ZeroDivisionError:
      #print("You can't normalize Zero Vector!")
      raise Exception("You can't normalize Zero Vector!")

  def angle_with(self, v, in_degrees=False):
    try:

      mag_1 = self.magnitude()
      mag_2 = v.magnitude()
      if in_degrees:
        return math.degrees(math.acos(self.dot_product_with(v)/(mag_1*mag_2)))
      else:
        return math.acos(self.dot_product_with(v)/(mag_1*mag_2))

    except ZeroDivisionError:
      #print("At least one of the vector is Zero Vector! No angle defined.")
      raise Exception("One of the vector is Zero Vector! No angle defined.")

  def is_zero_vector(self):
    return self.magnitude() < TOLERANCE
  
  def is_parallel_to(self, v):
    if v.is_zero_vector() or self.is_zero_vector():
      return True
    else:
      s_m = self.normalization()
      v_m = v.normalization()
      return s_m.minus(v_m).is_zero_vector() or s_m.plus(v_m).is_zero_vector() 

  def is_orthogonal_to(self, v):
    return abs(self.dot_product_with(v)) < TOLERANCE

  def component_parallel_to(self, base_vactor):
    u_b = base_vactor.normalization()
    mag = self.dot_product_with(u_b)
    return u_b.times_scalar(mag)

  def component_orthogonal_to(self, base_vactor):
    c_p = self.component_parallel_to(base_vactor)
    return self.minus(c_p)

  
  def cross_product_with(self, v):
    if self.dimension != 3 or v.dimension != 3:
      raise Exception("Both cross product vectors must be three dimensional")
    
    if self.is_parallel_to(v):
      return Vector([0 for i in range(self.dimension)])
    else:
      x = self.coordinates[1]*v.coordinates[2] - self.coordinates[2]*v.coordinates[1]
      y = self.coordinates[2]*v.coordinates[0] - self.coordinates[0]*v.coordinates[2]
      z = self.coordinates[0]*v.coordinates[1] - self.coordinates[1]*v.coordinates[0]
      return Vector([x, y, z])

  def area_of_parallelogram_spanned_with(self, v):
    new_self = self
    new_v = v
    if self.dimension == 2:
      new_self = Vector(self.coordinates + (0,))
    if v.dimension == 2:
      new_v = Vector(v.coordinates + (0,))

    return new_self.cross_product_with(new_v).magnitude()
  
  def area_of_triangle_spanned_with(self, v):
    return self.area_of_parallelogram_spanned_with(v) / Decimal(2)


















'''
v0 = Vector([5, 3, -2])
u0 = Vector([-1, 0, 3])
print(v0.cross_product_with(u0))

v1 = Vector([8.462, 7.893, -8.187])
v2 = Vector([6.984, -5.975, 4.778])
print(v1.cross_product_with(v2))

v3 = Vector([-8.987, -9.838, 5.031])
v4 = Vector([-4.268, -1.861, -8.866])
print(v3.area_of_parallelogram_spanned_with(v4))

v5 = Vector([1.5, 9.547, 3.691])
v6 = Vector([-6.007, 0.124, 5.772])
print(v5.area_of_triangle_spanned_with(v6))
'''

'''
v1 = Vector([3.039, 1.879])
base_1 = Vector([0.825, 2.036])

v2 = Vector([-9.88, -3.264, -8.159])
base_2 = Vector([-2.155, -9.353, -9.473])

v3 = Vector([3.009, -6.172, 3.692, -2.51])
#v3 = Vector([0, 0, 0, 0])
base_3 = Vector([6.404, -9.144, 2.759, 8.718])
#base_3 = Vector([0, 0, 0, 0])

print(v1.component_parallel_to(base_1))
print(v2.component_orthogonal_to(base_2))
print(v3.component_parallel_to(base_3))
print(v3.component_orthogonal_to(base_3))
'''



'''
v1 = Vector([-7.579, -7.88])
v2 = Vector([22.737, 23.64])

v3 = Vector([-2.029, 9.97, 4.172])
v4 = Vector([-9.231, -6.639, -7.245])

v5 = Vector([-2.328, -7.284, -1.214])
v6 = Vector([-1.821, 1.072, -2.94])

v7 = Vector([2.118, 4.827])
v8 = Vector([0, 0])


print(v1.is_parallel_with(v2))
print(v1.is_orthogonal_with(v2))

print(v3.is_parallel_with(v4))
print(v3.is_orthogonal_with(v4))

print(v5.is_parallel_with(v6))
print(v5.is_orthogonal_with(v6))

print(v7.is_parallel_with(v8))
print(v7.is_orthogonal_with(v8))

'''


'''
v1 = Vector([7.887, 4.138])
v2 = Vector([-8.802, 6.776])

v3 = Vector([-5.955, -4.904, -1.874])
v4 = Vector([-4.496, -8.755, 7.103])

v5 = Vector([3.183, -7.627])
v6 = Vector([-2.668, 5.319])

v7 = Vector([7.35, 0.221, 5.188])
v8 = Vector([2.751, 8.259, 3.985])


print(v1.dot_product_with(v2))
print(v3.dot_product_with(v4))
print(v5.angle_with(v6))
print(v7.angle_with(v8, in_degrees=True))
print(v5.angle_with(v5))
'''


'''
v1 = Vector([-0.221, 7.437])
v2 = Vector([8.813, -1.331, -6.247])

v3 = Vector([5.581, -2.136])
v4 = Vector([1.996, 3.108, -4.554])

v5 = Vector([0])
print(v1.magnitude())
print(v2.magnitude())
print(v3.normalization())
print(v4.normalization())
#print(v5.normalization())
'''