from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
import numpy as np
from functools import partial
import copy_reg
import types


# use copy_reg to make the instance method picklable,
# because multiprocessing must pickle things to sling them among process
def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)


copy_reg.pickle(types.MethodType, _pickle_method)



class A(object):
    def __init__(self):
        self.value = [0,0,0,0]
        self.array = np.zeros((4,))


class B(object):
    def change(self,a,j,i):
        a.value[i] = i
        a.array[i] = i
        j += 1

    def mp(self, a):
        func = partial(self.change, a, 10)
        pool = Pool()
        pool.map(func, [0,1,2,3])
        pool.close()

    def mt(self, a):
        func = partial(self.change, a, 10)
        pool = ThreadPool()
        pool.map(func, [0, 1, 2, 3])
        pool.close()



a = A()
b = B()
print a.value, a.array
b.mp(a)
print a.value, a.array
b.mt(a)
print a.value, a.array

# multiprocessing, a is copy for each processing, not share memory. this will cause high memory consumption
# multi threading, a is shared, but it is not real parallel
