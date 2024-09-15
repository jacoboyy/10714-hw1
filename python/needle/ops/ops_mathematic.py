"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy as array_api

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

BACKEND = "np"

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a ** b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return out_grad * rhs * power(lhs, rhs-1), out_grad * power(lhs, rhs) * log(lhs)

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a ** self.scalar

    def gradient(self, out_grad, node):
        return out_grad * self.scalar * power_scalar(node.inputs[0], self.scalar - 1)


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return out_grad / rhs, -out_grad * lhs / power_scalar(rhs, 2)

def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if not self.axes:
            return array_api.swapaxes(a, len(a.shape)-2, len(a.shape)-1)
        else:
            return array_api.swapaxes(a, self.axes[0], self.axes[1])

    def gradient(self, out_grad, node):
        return transpose(out_grad, self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return a.reshape(self.shape)

    def gradient(self, out_grad, node):
        return reshape(out_grad, node.inputs[0].shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ori_shape = node.inputs[0].shape # shape before broadcasting
        result = out_grad
        # handle dimension size mismatch
        for _ in range(len(self.shape) - len(ori_shape)):
            result = result.sum(axes=(0,))
        # squeeze broadcasted axis
        brd_shape = result.shape
        assert(len(brd_shape) == len(ori_shape))
        for idx in range(len(ori_shape)):
            ori_dim = ori_shape[idx]
            brd_dim = brd_shape[idx]
            if ori_dim != brd_dim:
                result = result.sum(axes=(idx,), keepdims=True)
        return result



def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None, keepdims=False):
        self.axes = axes
        self.keepdims = keepdims

    def compute(self, a):
        return array_api.sum(a, axis=self.axes, keepdims=self.keepdims)

    def gradient(self, out_grad, node):
        result = out_grad
        orig_shape = node.inputs[0].shape
        # reshape to bring back squeezed dimension
        new_shape = []
        for dim, size in enumerate(orig_shape):
            if not self.axes or dim in self.axes:
                new_shape.append(1)
            else:
                new_shape.append(size)
        result = out_grad.reshape(new_shape)
        # broadcast
        return broadcast_to(result, orig_shape)

def summation(a, axes=None, keepdims=False):
    return Summation(axes, keepdims)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return array_api.matmul(a, b)

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        lhs_grad = matmul(out_grad, transpose(rhs))
        rhs_grad = matmul(transpose(lhs), out_grad)
        # handle dimension mismatch
        for _ in range(len(lhs_grad.shape) - len(node.inputs[0].shape)):
            lhs_grad = lhs_grad.sum(0)
        for _ in range(len(rhs_grad.shape) - len(node.inputs[1].shape)):
            rhs_grad = rhs_grad.sum(0)
        return lhs_grad, rhs_grad


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return -out_grad


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        return out_grad / node.inputs[0]


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        return multiply(out_grad, exp(node.inputs[0]))

def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(0, a)

    def gradient(self, out_grad, node):
        output = node.realize_cached_data()
        flag = (output > 0).astype(array_api.int8)
        flag_tensor = Tensor(flag, requires_grad=False)
        return out_grad * flag_tensor

def relu(a):
    return ReLU()(a)
