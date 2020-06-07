# TODO: make test_element_wise slim (#1055)
import taichi as ti
import numpy as np
from taichi import allclose
import pytest


def _c_mod(a, b):
    return a - b * int(float(a) / b)


@pytest.mark.parametrize('is_mat', [(True, True), (True, False),
                                    (False, True)])
@ti.all_archs
def test_binary_f(is_mat):
    lhs_is_mat, rhs_is_mat = is_mat

    x = ti.Matrix(3, 2, ti.f32, 16)
    if lhs_is_mat:
        y = ti.Matrix(3, 2, ti.f32, ())
    else:
        y = ti.var(ti.f32, ())
    if rhs_is_mat:
        z = ti.Matrix(3, 2, ti.f32, ())
    else:
        z = ti.var(ti.f32, ())

    if lhs_is_mat:
        y.from_numpy(np.array([[0, 2], [9, 3], [7, 4]], np.float32))
    else:
        y[None] = 6
    if rhs_is_mat:
        z.from_numpy(np.array([[4, 5], [6, 3], [9, 2]], np.float32))
    else:
        z[None] = 5

    @ti.kernel
    def func():
        x[0] = y[None] + z[None]
        x[1] = y[None] - z[None]
        x[2] = y[None] * z[None]
        x[3] = y[None] / z[None]
        x[4] = y[None] // z[None]
        x[5] = y[None] % z[None]
        x[6] = y[None]**z[None]
        x[7] = y[None] == z[None]
        x[8] = y[None] != z[None]
        x[9] = y[None] > z[None]
        x[10] = y[None] >= z[None]
        x[11] = y[None] < z[None]
        x[12] = y[None] <= z[None]
        x[13] = ti.atan2(y[None], z[None])
        x[14] = ti.min(y[None], z[None])
        x[15] = ti.max(y[None], z[None])

    func()
    x = x.to_numpy()
    y = y.to_numpy()
    z = z.to_numpy()
    assert allclose(x[0], y + z)
    assert allclose(x[1], y - z)
    assert allclose(x[2], y * z)
    assert allclose(x[3], y / z)
    assert allclose(x[4], y // z)
    assert allclose(x[5], y % z)
    assert allclose(x[6], y**z)
    assert allclose(x[7], y == z)
    assert allclose(x[8], y != z)
    assert allclose(x[9], y > z)
    assert allclose(x[10], y >= z)
    assert allclose(x[11], y < z)
    assert allclose(x[12], y <= z)
    assert allclose(x[13], np.arctan2(y, z))
    assert allclose(x[14], np.minimum(y, z))
    assert allclose(x[15], np.maximum(y, z))


@pytest.mark.parametrize('is_mat', [(True, True), (True, False),
                                    (False, True)])
@ti.all_archs
def test_binary_i(is_mat):
    lhs_is_mat, rhs_is_mat = is_mat

    x = ti.Matrix(3, 2, ti.i32, 19)
    if lhs_is_mat:
        y = ti.Matrix(3, 2, ti.i32, ())
    else:
        y = ti.var(ti.i32, ())
    if rhs_is_mat:
        z = ti.Matrix(3, 2, ti.i32, ())
    else:
        z = ti.var(ti.i32, ())

    if lhs_is_mat:
        y.from_numpy(np.array([[0, 2], [9, 3], [7, 4]], np.int32))
    else:
        y[None] = 6
    if rhs_is_mat:
        z.from_numpy(np.array([[4, 5], [6, 3], [9, 2]], np.int32))
    else:
        z[None] = 5

    @ti.kernel
    def func():
        x[0] = y[None] + z[None]
        x[1] = y[None] - z[None]
        x[2] = y[None] * z[None]
        x[3] = y[None] // z[None]
        x[4] = ti.raw_div(y[None], z[None])
        x[5] = y[None] % z[None]
        x[6] = ti.raw_mod(y[None], z[None])
        x[7] = y[None]**z[None]
        x[8] = y[None] == z[None]
        x[9] = y[None] != z[None]
        x[10] = y[None] > z[None]
        x[11] = y[None] >= z[None]
        x[12] = y[None] < z[None]
        x[13] = y[None] <= z[None]
        x[14] = y[None] & z[None]
        x[15] = y[None] ^ z[None]
        x[16] = y[None] | z[None]
        x[17] = ti.min(y[None], z[None])
        x[18] = ti.max(y[None], z[None])

    func()
    x = x.to_numpy()
    y = y.to_numpy()
    z = z.to_numpy()
    assert allclose(x[0], y + z)
    assert allclose(x[1], y - z)
    assert allclose(x[2], y * z)
    assert allclose(x[3], y // z)
    assert allclose(x[4], y // z)
    assert allclose(x[5], y % z)
    assert allclose(x[6], y % z)
    assert allclose(x[7], y**z)
    assert allclose(x[8], y == z)
    assert allclose(x[9], y != z)
    assert allclose(x[10], y > z)
    assert allclose(x[11], y >= z)
    assert allclose(x[12], y < z)
    assert allclose(x[13], y <= z)
    assert allclose(x[14], y & z)
    assert allclose(x[15], y ^ z)
    assert allclose(x[16], y | z)
    assert allclose(x[17], np.minimum(y, z))
    assert allclose(x[18], np.maximum(y, z))


@pytest.mark.parametrize('rhs_is_mat', [True, False])
@ti.all_archs
def test_writeback_binary_f(rhs_is_mat):
    x = ti.Matrix(3, 2, ti.f32, 9)
    y = ti.Matrix(3, 2, ti.f32, ())
    if rhs_is_mat:
        z = ti.Matrix(3, 2, ti.f32, ())
    else:
        z = ti.var(ti.f32, ())

    y.from_numpy(np.array([[0, 2], [9, 3], [7, 4]], np.float32))
    if rhs_is_mat:
        z.from_numpy(np.array([[4, 5], [6, 3], [9, 2]], np.float32))
    else:
        z[None] = 5

    @ti.kernel
    def func():
        for i in x:
            x[i] = y[None]
        if ti.static(rhs_is_mat):
            x[0] = z[None]
        else:
            x[0].fill(z[None])
        x[1] += z[None]
        x[2] -= z[None]
        x[3] *= z[None]
        x[4] /= z[None]
        x[5] //= z[None]
        x[6] %= z[None]
        ti.atomic_min(x[7], z[None])
        ti.atomic_max(x[8], z[None])

    func()
    x = x.to_numpy()
    y = y.to_numpy()
    z = z.to_numpy()
    assert allclose(x[1], y + z)
    assert allclose(x[2], y - z)
    assert allclose(x[3], y * z)
    assert allclose(x[4], y / z)
    assert allclose(x[5], y // z)
    assert allclose(x[6], y % z)
    assert allclose(x[7], np.minimum(y, z))
    assert allclose(x[8], np.maximum(y, z))


@pytest.mark.parametrize('rhs_is_mat', [(True, True), (True, False)])
@ti.all_archs
def test_writeback_binary_i(rhs_is_mat):
    x = ti.Matrix(3, 2, ti.i32, 12)
    y = ti.Matrix(3, 2, ti.i32, ())
    if rhs_is_mat:
        z = ti.Matrix(3, 2, ti.i32, ())
    else:
        z = ti.var(ti.i32, ())

    y.from_numpy(np.array([[0, 2], [9, 3], [7, 4]], np.int32))
    if rhs_is_mat:
        z.from_numpy(np.array([[4, 5], [6, 3], [9, 2]], np.int32))
    else:
        z[None] = 5

    @ti.kernel
    def func():
        for i in x:
            x[i] = y[None]
        x[0] = z[None]
        x[1] += z[None]
        x[2] -= z[None]
        x[3] *= z[None]
        x[4] //= z[None]
        x[5] %= z[None]
        x[6] &= z[None]
        x[7] |= z[None]
        x[8] ^= z[None]
        ti.atomic_min(x[10], z[None])
        ti.atomic_max(x[11], z[None])

    func()
    x = x.to_numpy()
    y = y.to_numpy()
    z = z.to_numpy()
    assert allclose(x[1], y + z)
    assert allclose(x[2], y - z)
    assert allclose(x[3], y * z)
    assert allclose(x[4], y // z)
    assert allclose(x[5], y % z)
    assert allclose(x[6], y & z)
    assert allclose(x[7], y | z)
    assert allclose(x[8], y ^ z)
    assert allclose(x[10], np.minimum(y, z))
    assert allclose(x[11], np.maximum(y, z))


@ti.all_archs
def test_unary():
    xi = ti.Matrix(3, 2, ti.i32, 4)
    yi = ti.Matrix(3, 2, ti.i32, ())
    xf = ti.Matrix(3, 2, ti.f32, 13)
    yf = ti.Matrix(3, 2, ti.f32, ())

    yi.from_numpy(np.array([[3, 2], [9, 0], [7, 4]], np.int32))
    yf.from_numpy(np.array([[0.3, 0.2], [0.9, 0.1], [0.7, 0.4]], np.float32))

    @ti.kernel
    def func():
        xi[0] = -yi[None]
        xi[1] = ~yi[None]
        xi[2] = ti.logical_not(yi[None])
        xi[3] = ti.abs(yi[None])
        xf[0] = -yf[None]
        xf[1] = ti.abs(yf[None])
        xf[2] = ti.sqrt(yf[None])
        xf[3] = ti.sin(yf[None])
        xf[4] = ti.cos(yf[None])
        xf[5] = ti.tan(yf[None])
        xf[6] = ti.asin(yf[None])
        xf[7] = ti.acos(yf[None])
        xf[8] = ti.tanh(yf[None])
        xf[9] = ti.floor(yf[None])
        xf[10] = ti.ceil(yf[None])
        xf[11] = ti.exp(yf[None])
        xf[12] = ti.log(yf[None])

    func()
    xi = xi.to_numpy()
    yi = yi.to_numpy()
    xf = xf.to_numpy()
    yf = yf.to_numpy()
    assert allclose(xi[0], -yi)
    assert allclose(xi[1], ~yi)
    assert allclose(xi[3], np.abs(yi))
    assert allclose(xf[0], -yf)
    assert allclose(xf[1], np.abs(yf))
    assert allclose(xf[2], np.sqrt(yf))
    assert allclose(xf[3], np.sin(yf))
    assert allclose(xf[4], np.cos(yf))
    assert allclose(xf[5], np.tan(yf))
    assert allclose(xf[6], np.arcsin(yf))
    assert allclose(xf[7], np.arccos(yf))
    assert allclose(xf[8], np.tanh(yf))
    assert allclose(xf[9], np.floor(yf))
    assert allclose(xf[10], np.ceil(yf))
    assert allclose(xf[11], np.exp(yf))
    assert allclose(xf[12], np.log(yf))