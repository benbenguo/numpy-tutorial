# -*- coding: utf-8 -*-

import numpy as np

'''
NumPy是一个功能强大的Python库，主要用于对多维数组执行计算。
NumPy这个词来源于两个单词-- Numerical和Python。
NumPy提供了大量的库函数和操作，可以帮助程序员轻松地进行数值计算。这类数值计算广泛用于以下任务：

 - 机器学习模型：在编写机器学习算法时，需要对矩阵进行各种数值计算。
   例如矩阵乘法、换位、加法等。
   NumPy提供了一个非常好的库，用于简单(在编写代码方面)和快速(在速度方面)计算。
   NumPy数组用于存储训练数据和机器学习模型的参数。

 - 图像处理和计算机图形学：计算机中的图像表示为多维数字数组。
   NumPy成为同样情况下最自然的选择。
   实际上，NumPy提供了一些优秀的库函数来快速处理图像。
   例如，镜像图像、按特定角度旋转图像等。

 - 数学任务：NumPy对于执行各种数学任务非常有用，
   如数值积分、微分、内插、外推等。因此，当涉及到数学任务时，它形成了一种基于Python的MATLAB的快速替代。
'''

"""
快速定义一维NumPy数组
在上面的简单示例中，我们首先使用import numpy作为np导入NumPy库。
然后，我们创建了一个包含5个整数的简单NumPy数组，然后我们将其打印出来。
"""

my_array = np.array([1, 2, 3, 4, 5])

print("my_array = np.array([1, 2, 3, 4, 5])")
print(my_array, "\n")

"""
打印我们创建的数组的形状：(5, )。意思就是 my_array 是一个包含5个元素的数组。
"""

print("my_array.shape")
print(my_array.shape, "\n")

"""
打印各个元素，NumPy数组的起始索引编号为0。
"""
print("my_array[0]")
print(my_array[0], "\n")

print("my_array[1]")
print(my_array[1], "\n")

"""
修改NumPy数组的元素。
"""
my_array[0] = -1
print("my_array[0] = -1")
print(my_array, "\n")

"""
要创建一个长度为5的NumPy数组，但所有元素都为0
"""
my_new_array = np.zeros(5)
print("my_new_array = np.zeros(5)")
print(my_new_array, "\n")

"""
创建一个随机值数组
"""
my_random_array = np.random.random(5)
print("my_random_array = np.random.random(5)")
print(my_random_array, "\n")

"""
要创建一个长度为5的NumPy数组，但所有元素都为1
"""
my_ones_array = np.ones(5)
print("my_ones_array = np.ones(5)")
print(my_ones_array, "\n")

"""
使用NumPy创建二维数组。try it np.zeros((2, 5)), what's the different with np.zeros(5)?
"""
my_2d_zeros_array = np.zeros((2, 5))
print("my_2d_zeros_array = np.zeros((2, 5))")
print(my_2d_zeros_array, "\n")

my_2d_ones_array = np.ones((2, 5))
print("my_2d_ones_array = np.ones((2, 5))")
print(my_2d_ones_array, "\n")

my_array = np.array([[4, 5],
                     [6, 1]])
print("my_array[0][1]")
print(my_array[0][1], "\n")

print("my_array.shape")
print(my_array.shape, "\n")

"""
NumPy提供了一种提取多维数组的行/列的强大方法。
想从中提取第二列（索引1）的所有元素。
"""
my_array_column_2 = my_array[:, 1]
print("my_array_column_2 = my_array[:, 1]")
print(my_array_column_2, "\n")

"""
使用NumPy，你可以轻松地在数组上执行数学运算。
乘法运算符执行逐元素乘法而不是矩阵乘法。
"""
a = np.array([[1.0, 2.0],
              [3.0, 4.0]])

b = np.array([[5.0, 6.0],
              [7.0, 8.0]])

ab_sum = a + b
ab_difference = a - b
ab_product = a * b
ab_quotient = a / b

print("ab_sum = a + b")
print("Sum = ", ab_sum, "\n")

print("ab_difference = a - b")
print("Difference = ", ab_difference, "\n")

print("ab_product = a * b")
print("Product = ", ab_product, "\n")

print("ab_quotient = a / b")
print("Quotient = ", ab_quotient, "\n")

# The output will be as follows:

# >> Sum = [[ 6. 8.] [10. 12.]]
# >> Difference = [[-4. -4.] [-4. -4.]]
# >> Product = [[ 5. 12.] [21. 32.]]
# >> Quotient = [[0.2 0.33333333] [0.42857143 0.5 ]]

"""
执行矩阵乘法。
"""
matrix_product = a.dot(b)
print("matrix_product = a.dot(b)")
print("Matrix Product = ", matrix_product, "\n")
