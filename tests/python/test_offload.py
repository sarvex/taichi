import taichi as ti
from tests import test_utils


@test_utils.test()
def test_running_loss():
    return


@test_utils.test()
def test_reduce_separate():
    a = ti.field(ti.f32, shape=(16))
    b = ti.field(ti.f32, shape=(4))
    c = ti.field(ti.f32, shape=())

    ti.root.lazy_grad()

    @ti.kernel
    def reduce1():
        for i in range(16):
            b[i // 4] += a[i]

    @ti.kernel
    def reduce2():
        for i in range(4):
            c[None] += b[i]

    c.grad[None] = 1
    reduce2.grad()
    reduce1.grad()

    for i in range(4):
        assert b.grad[i] == 1
    for i in range(16):
        assert a.grad[i] == 1


@test_utils.test()
def test_reduce_merged():
    a = ti.field(ti.f32, shape=(16))
    b = ti.field(ti.f32, shape=(4))
    c = ti.field(ti.f32, shape=())

    ti.root.lazy_grad()

    @ti.kernel
    def reduce():
        for i in range(16):
            b[i // 4] += a[i]

        for i in range(4):
            c[None] += b[i]

    c.grad[None] = 1
    reduce.grad()

    for i in range(4):
        assert b.grad[i] == 1
    for i in range(16):
        assert a.grad[i] == 1
