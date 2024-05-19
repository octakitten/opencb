# 32 params in this object that we need

# for each dimension asked for, create a one of these that has a dimension as
# the object in either of its rows
#
# at least thats what it'd do in a loop, sort of
# its much more abstract here, not sure how tf this works
# guess we'll find out

def layers():
  return 32

def max_pixel():
  return +255
      
type u24_overflow:
  data { n }

type bool:
  true
  false
  overflow

def bool/not(b):
  fold b:
    case bool/true:
      return bool/false
    case bool/false:
      return bool/true
    case bool/overflow:
      return bool/overflow

def bool/num(n):
  fold n:
    case 0:
      return bool/false
    case u24_overflow/data:
      return bool/overflow
    case _:
      return bool/true

type matrix:
  Nil
  Cons { ~this, ~next }

type vector:
  Nil
  Cons { this, ~next }

def step(n):
  switch n:
    case 0:
      return 0
    case _:
      return n-1

def decr(n):
  match bool/num(n):
    case bool/true:
      return n-1
    case bool/false:
      return u24_overflow/data { n: 0 }

def greater(n1, n2):
  match n1:
    case u24_overflow/data:
      return bool/false
    case 0:
      return bool/not(bool/num(n2))
    case _:
      return greater(decr(n1), decr(n2))

def greater_equals(n1, n2):
  match n1:
    case u24_overflow/data:
      return check_overflow(n2)
    case 0:
      return bool/not(bool/num(n2))
    case _:
      return greater_equals(decr(n1), decr(n2))
    
def is_overflow(n):
  match n:
    case bool/overflow:
      return bool/true
    case u24_overflow/data:
      return bool/true
    case _:
      return bool/false

def vector/at(v, n):
  open vector: v
  switch n:
    case 0:
      return v.this
    case _:
      return vector/at(v, n-1)

def addif(v1, v2, v3):
  match v1:
    case bool/true:
      return v2 + v3
    case _:
      return 0

def matrix/at(m, n):
  open matrix: m
  switch n:
    case 0:
      return m.this
    case _:
      return matrix/at(m, n-1)

def gen_zeros(n):
  switch n:
    case 0:
      return vector/Cons { this: +0 , next: vector/Nil }
    case _:
      return vector/Cons {  this: +0 , next: gen_zeros(n-1) }

def gen_size(n, s):
  switch s:
    case 0:
      return matrix/Cons { this: gen_zeros(s), next: matrix/Nil }
    case _:
      return matrix/Cons { this: gen_size(s), next:gen_size(s-1) }

def gen_dims(n, s):
  switch n:
    case 0:
      return matrix/Cons { this: gen_size(s), next: matrix/Nil }
    case _:
      return matrix/Cons { this: gen_dims(n, s), next: gen_dims(n-1, s) }

def gen_matrix(dims, size):
  switch dims:
    case 0:
      return matrix/Cons { this: gen_zeros(size), next: matrix/Nil }
    case _:
      return matrix/Cons { this: gen_matrix(dims-1, size), next: gen_matrix(dims-1, size) }

def add_pixel(m, v):
  open matrix: m
  open vector: v
  match v:
    

def make_image(size):
  return gen_dims(2, size)

def make_starting_positions(image, size):
  len1 = size / 4
  len2 = size * 3 / 4
  v1 = vector/Cons { this: len1, next: vector/Cons{ this: len1, next: vector/Nil } }
  v2 = vector/Cons { this: len1, next: vector/Cons{ this: len2, next: vector/Nil } }
  v3 = vector/Cons { this: len2, next: vector/Cons{ this: len1, next: vector/Nil } }
  v4 = vector/Cons { this: len2, next: vector/Cons{ this: len2, next: vector/Nil } }
  add_pixel(matrix/at(image, step(3)), v1)
  add_pixel(matrix/at(image, step(3)), v2)
  add_pixel(matrix/at(image, step(3)), v3)
  add_pixel(matrix/at(image, step(3)), v4)
  return image

def update(model, d, s, b, image):
  # add image to layer1
  layer1 = vector/at(matrix/at(model, step(s)), vector/Cons { this: 1, next: vector/Nil })
  layer0 = layer0 + layer1
  layer1 = layer1 / 2
  positive_firing = greater(layer0, layer3)
  negative_firing = greater(layer4, layer0)
  layer0 = addif(positive_firing, layer0, layer5)
  layer0 = addif(negative_firing, layer0, layer6)
  layer3 = 



def main(dims, size, bounds, controls, senses):
  return make_model(dims, size, bounds, controls, senses)

def make_model( dims, size, bounds, complexity ):
  d = dims
  s = size
  b = bounds
  c = complexity
  # make the model 
  # we can probably call this inline at some point
  model = gen_dims(d, s)
  input_image = make_image(s)
  starting_images = make_starting_positions(input_image, s)
  # vector/at(matrix/at(model, step(s)), step(layers()+(d*c))
  


  return 0


  # then we have to start iterating on it somehow
  # we need an image array, but we have to create it ourselves in here
  # we're locked inside the bend interpreter so...
  # we'll run through all possible versions of the starting image
  # and in each of those we'll increment or decrement a value in the model
  # only one at a time though... somehow?
  # but actually, now that i think about it
  # it might be better to create an image with maybe... 4 versions of it
  # it'll just be a vector matrix so we'll represent it with a bend, easy enough
  # we can get the four versions of the image to be the four points on the image
  # that are all roughly equidistance between the corner and the center
  # then we'll test the model at each version of it
  # with each of those four images
  # if the model can move the dot closer to the edge, cool, if not, revert and 
  # try a different value
  # 
  # that's what the flow is supposed to look like anyway

