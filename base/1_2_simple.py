# -*- coding: utf-8 -*-

import numpy as np

"""
数组基础

创建一个数组

NumPy围绕这些称为数组的事物展开。
实际上它被称之为 ndarrays，你不知道没事儿。
使用NumPy提供的这些数组，我们就可以以闪电般的速度执行各种有用的操作，
如矢量和矩阵、线性代数等数学运算！
（开个玩笑，本文章中我们不会做任何繁重的数学运算）
"""

# 1D Array
a = np.array([0, 1, 2, 3, 4])
b = np.array((0, 1, 2, 3, 4))
# 0至4，step = 1，5个元素
c = np.arange(5)
# 0至2*np.pi，均匀分布，5个元素
d = np.linspace(0, 2*np.pi, 5)

print("a:")
print(a, "\n")  # >>>[0 1 2 3 4]

print("b:")
print(b, "\n")  # >>>[0 1 2 3 4]

print("c:")
print(c, "\n")  # >>>[0 1 2 3 4]

print("d:")
print(d, "\n")  # >>>[ 0.          1.57079633  3.14159265  4.71238898  6.28318531]

print("a[3]")
print(a[3], "\n")  # >>>3


"""
上面的数组示例是如何使用NumPy表示向量的，
接下来我们将看看如何使用多维数组表示矩阵和更多的信息。
"""

# MD Array,
a = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])

print("a[2, 4]")
print(a[2, 4], "\n")  # >>>25

"""
多维数组切片

切片多维数组比1D数组复杂一点，
并且在使用NumPy时你也会经常需要使用到。
"""

# MD slicing
print("a[0, 1:4]")
print(a[0, 1:4], "\n")    # >>>[12 13 14]

print("a[1:4, 0]")
print(a[1:4, 0], "\n")    # >>>[16 21 26]

print("a[::2, ::2]")
print(a[::2, ::2], "\n")  # >>>[[11 13 15]
                          #     [21 23 25]
                          #     [31 33 35]]
print("a[:, 1]")
print(a[:, 1], "\n")      # >>>[12 17 22 27 32]

"""
数组属性

在使用 NumPy 时，你会想知道数组的某些信息。
很幸运，在这个包里边包含了很多便捷的方法，可以给你想要的信息。
"""

# Array properties
a = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])

print("type(a)")
print(type(a), "\n")  # >>><class 'numpy.ndarray'>

print("a.dtype")
print(a.dtype, "\n")  # >>>int64

print("a.size")
print(a.size, "\n")  # >>>25

print("a.shape")
print(a.shape, "\n")  # >>>(5, 5)

# itemsize 属性是每个项占用的字节数。
print("a.itemsize")
print(a.itemsize, "\n")  # >>>8

# ndim 属性是数组的维数。这个有2个。
print("a.ndim")
print(a.ndim, "\n")  # >>>2

# nbytes 属性是数组中的所有数据消耗掉的字节数。
# 你应该注意到，这并不计算数组的开销，因此数组占用的实际空间将稍微大一点。
print("a.nbytes")
print(a.nbytes, "\n")  # >>>200


"""
使用数组
基本操作符

只是能够从数组中创建和检索元素和属性不能满足你的需求，
你有时也需要对它们进行数学运算。 
你完全可以使用四则运算符 +、- 、/ 来完成运算操作。

除了 dot() 之外，这些操作符都是对数组进行逐元素运算。
比如 (a, b, c) + (d, e, f) 的结果就是 (a+d, b+e, c+f)。
它将分别对每一个元素进行配对，然后对它们进行运算。
它返回的结果是一个数组。

注意，当使用逻辑运算符比如 “<” 和 “>” 的时候，
返回的将是一个布尔型数组，这点有一个很好的用处，后边我们会提到。

dot() 函数计算两个数组的点积。
它返回的是一个标量（只有大小没有方向的一个值）而不是数组。

点积
两个向量a = [a1, a2,…, an]和b = [b1, b2,…, bn]的点积定义为：
a·b = a1 * b1 + a2 * b2 + ... + an * bn
"""

# Basic Operators
a = np.arange(25)
a = a.reshape((5, 5))

print("a:")
print(a, "\n")

b = np.array([10, 62, 1, 14, 2, 56, 79, 2, 1, 45,
              4, 92, 5, 55, 63, 43, 35, 6, 53, 24,
              56, 3, 56, 44, 78])
b = b.reshape((5, 5))

print("b:")
print(b, "\n")

print("a + b")
print(a + b, "\n")

print("a - b")
print(a - b, "\n")

print("a * b")
print(a * b, "\n")

print("a / b")
print(a / b, "\n")

print("a ** 2")
print(a ** 2, "\n")

print("a < b")
print(a < b, "\n")

print("a > b")
print(a > b, "\n")

print("a.dot(b)")
print(a.dot(b), "\n")

"""
数组特殊运算符

NumPy还提供了一些别的用于处理数组的好用的运算符。

sum()、min()和max()函数的作用非常明显。
将所有元素相加，找出最小和最大元素。
"""

# dot, sum, min, max, cumsum
a = np.arange(10)

print("a:")
print(a, "\n")

print("a.sum()")
print(a.sum(), "\n")  # >>>45

print("a.min()")
print(a.min(), "\n")  # >>>0

print("a.max()")
print(a.max(), "\n")  # >>>9

"""
然而，cumsum()函数就不那么明显了。
它将像sum()这样的每个元素相加，
但是它首先将第一个元素和第二个元素相加，
并将计算结果存储在一个列表中，然后将该结果添加到第三个元素中，
然后再将该结果存储在一个列表中。
这将对数组中的所有元素执行此操作，并返回作为列表的数组之和的运行总数。
"""

print("a.cumsum()")
print(a.cumsum(), "\n")  # >>>[ 0  1  3  6 10 15 21 28 36 45]


