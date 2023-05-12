import pytest

import taichi as ti
from tests import test_utils


@test_utils.test()
def test_if():
    x = ti.field(ti.f32)

    ti.root.dense(ti.i, 1).place(x)

    @ti.kernel
    def func():
        a = 0 if True else 1
        print(a)

    with pytest.raises(Exception):
        func()


@test_utils.test()
def test_for():
    x = ti.field(ti.f32)

    ti.root.dense(ti.i, 1).place(x)

    @ti.kernel
    def func():
        for i in range(10):
            a = i
        print(a)

    with pytest.raises(Exception):
        func()


@test_utils.test()
def test_while():
    x = ti.field(ti.f32)

    ti.root.dense(ti.i, 1).place(x)

    @ti.kernel
    def func():
        while True:
            a = 0
        print(a)

    with pytest.raises(Exception):
        func()
