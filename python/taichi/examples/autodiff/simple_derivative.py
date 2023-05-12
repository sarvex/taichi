import matplotlib.pyplot as plt

import taichi as ti

ti.init(arch=ti.cpu)

N = 2048
x, y = ti.field(ti.f32), ti.field(ti.f32)

ti.root.dense(ti.i, N).place(x, x.grad, y, y.grad)


@ti.kernel
def poly():
    guard = 0.2
    for i in x:
        v = x[i]
        ret = 0.0
        ret = 4 / ti.max(v, 0.1) if v < -guard or v > guard else 0
        y[i] = ret


xs = []
ys = []
for i in range(N):
    v = ((i + 0.5) / N) * 2 - 1
    xs.append(v)
    x[i] = v

poly()

print('y')
for i in range(N):
    y.grad[i] = 1
    ys.append(y[i])
print()

poly.grad()
print('grad_x')
grad_xs = [x.grad[i] for i in range(N)]
plt.title('Auto Diff')
ax = plt.gca()
ax.plot(xs, ys, label='f(x)')
ax.plot(xs, grad_xs, label='f\'(x)')
ax.legend()
ax.grid(True)
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
plt.show()
