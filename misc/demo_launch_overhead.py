import time

import taichi as ti

ti.init()


@ti.kernel
def compute_div(a: ti.i32):
    pass


compute_div(0)
print("starting...")
t = time.time()
for _ in range(100000):
    compute_div(0)
print((time.time() - t) * 10, 'us')
exit(0)
