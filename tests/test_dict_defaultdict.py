from collections import defaultdict
from time import time

d1 = {}
d2 = defaultdict(lambda : 0)

for i in range(100000,200000):
    d1[i] = i
    d2[i] = i

t1 = time()
for i in range(100000):
    v = i%2
    if v in d1:
        d1[v] += 1
    else:
        d1[v] = 1
t2 = time()

for i in range(100000):
    v = i%2
    d2[v] += 1

t3 = time()

print t2-t1, t3-t2

