import ctypes
import math
import time

import taichi as ti

libm = ctypes.CDLL('libm.so.6')

x, y = ti.field(ti.f32), ti.field(ti.f32)


@ti.kernel
def laplace():
    for i, j in x:
        y[i,
          j] = 4.0 * x[i, j] - x[i - 1, j] - x[i + 1,
                                               j] - x[i, j - 1] - x[i, j + 1]


ti.root.dense(ti.ij, (16, 16)).place(x).place(y)

laplace()

t = time.time()
N = 1000000
for i in range(N):
    x[i & 7, i & 7] = 1.0
print((time.time() - t) / N * 1e9, 'ns')

t = time.time()
N = 1000000
a = sum(x[i, i] for i in range(N))
print((time.time() - t) / N * 1e9, 'ns')

t = time.time()
N = 1000000
sin = getattr(libm, 'sin')
a = sum(sin(i) for i in range(N))
print((time.time() - t) / N * 1e9, 'ns')

t = time.time()
N = 1000000
a = sum(math.sin(i) for i in range(N))
print((time.time() - t) / N * 1e9, 'ns')
