import math

class Vector(object):
  # 实例（instance）的创建
  def __init__(self, coordinates):
    try:
      if not coordinates:
          raise ValueError
      self.coordinates = tuple(coordinates)
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
    return math.sqrt(self.dot_product_with(self))

  def normalization(self):
    try:
      mag = self.magnitude()
      return self.times_scalar(1/mag)
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

  



v5 = Vector([-8.802, 6.776])
print(v5.magnitude())
print(v5.dot_product_with(v5))


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