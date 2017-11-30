Python之科学计算宝典
============================
<!-- markdown-toc start - Don't forget to edit this section according to your modifications -->
**目录**

- [Python之科学计算宝典](#Python之科学计算宝典)
    - [Python基础](#Python基础)
        - [类型](#类型)
        - [列表](#列表)
        - [字典](#字典)
        - [集合](#集合)
        - [字符串](#字符串)
        - [操作符](#操作符)
        - [控制流](#控制流)
        - [函数, 类, 生成器, 修饰器](#函数-类-生成器-修饰器)
    - [NumPy](#Numpy-import-numpy-as-np)
        - [数组初始化](#数组初始化)
        - [索引](#索引)
        - [数组属性和操作](#数组属性和操作)
        - [布尔数组](#布尔数组)
        - [元素间逐个计算操作和数学函数](#元素间逐个计算操作和数学函数)
        - [内/外积](#内/外积)
        - [线性代数/矩阵数学](#线性代数/矩阵数学)
        - [读/写文件](#读/写文件)
        - [插值, 积分, 优化](#插值-积分-优化)
        - [傅立叶变换](#傅立叶变换)
        - [取整](#取整)
        - [随机变量](#随机变量)
    - [Matplotlib](#matplotlib-import-matplotlib.pyplot-as-plt)
        - [图和坐标轴](#图和坐标轴)
        - [图和坐标轴属性](#图和坐标轴属性)
        - [画图函数](#画图函数)
    - [Scipy](#scipy-import-scipy-as-sci)
        - [插值](#插值)
        - [线性代数](#线性代数)
        - [积分](#积分)
    - [Pandas](#pandas-import-pandas-as-pd)
        - [数据结构](#数据结构)
        - [数据框](#数据框)
        
<!-- markdown-toc end -->

## Python基础

### 类型
```python
a = 2           # 整数
b = 5.0         # 浮点
c = 8.3e5       # 指数
d = 1.5 + 0.5j  # 复数
e = 4 > 5       # 布尔
f = 'word'      # 字符串
```

### 列表

```python
a = ['red', 'blue', 'green']       # 手动初始化
b = list(range(5))                 # 从迭代初始化
c = [nu**2 for nu in b]            # 列表理解(list comprehension)
d = [nu**2 for nu in b if nu < 3]  # 条件列表理解(conditioned list comprehension)
e = c[0]                           # 访问元素
f = c[1:2]                         # 访问列表的一个片段
g = c[-1]                          # 访问最后一个元素
h = ['re', 'bl'] + ['gr']          # 列表连接(list concatenation)
i = ['re'] * 5                     # 重复一个列表
['re', 'bl'].index('re')           # 返回're'的索引
a.append('yellow')                 # 将新元素添加到列表的末尾
a.extend(b)                        # 将列表“b”中的元素添加到列表“a”的末尾
a.insert(1, 'yellow')              # 将元素插入指定位置
're' in ['re', 'bl']               # 如果're'在列表中，则返回真
'fi' not in ['re', 'bl']           # 如果'fi'不在列表中，则返回真
sorted([3, 2, 1])                  # 返回排序列表
a.pop(2)                           # 删除并返回索引指向的元素（默认最后一个）
```

### 字典

```python
a = {'red': 'rouge', 'blue': 'bleu'}         # 字典
b = a['red']                                 # 获取指定键对应的值
'red' in a                                   # 如果字典a包含键'红'，则返回真
c = [value for key, value in a.items()]      # 遍历字典
d = a.get('yellow', 'no translation found')  # 返回默认值
a.setdefault('extra', []).append('cyan')     # 用默认值初始化键
a.update({'green': 'vert', 'brown': 'brun'}) # 用另一个字典里的数据更新当前字典
a.keys()                                     # 获取键列表
a.values()                                   # 获取值列表
a.items()                                    # 获取键-值对的列表
del a['red']                                 # 删除键和与其关联的值
a.pop('blue')                                # 删除指定的键并返回相应的值
```


### 集合

```python
a = {1, 2, 3}                                # 手动初始化
b = set(range(5))                            # 从可迭代对象初始化
a.add(13)                                    # 将新元素添加到集合
a.discard(13)                                # 从集合中舍弃元素
a.update([21, 22, 23])                       # 用来自可迭代对象中的元素更新当前集合
a.pop()                                      # 删除并返回一个任意的集合元素
2 in {1, 2, 3}                               # 如果集合中包含2，则返回真
5 not in {1, 2, 3}                           # 如果集合中不包含5，则返回真
a.issubset(b)                                # 测试集合a中的每个元素是否都在集合b中
a <= b                                       # 测试集合a中的每个元素是否都在集合b中（用操作符表示）
a.issuperset(b)                              # 测试集合b中的每个元素是否都在集合a中
a >= b                                       # 测试集合b中的每个元素是否都在集合a中（用操作符表示）
a.intersection(b)                            # 将两个集合的交集作为一个新集合返回
a.difference(b)                              # 将两个或多个集合的差集作为新集合返回
a - b                                        # 将两个或多个集合的差集作为新集合返回（用操作符表示）
a.symmetric_difference(b)                    # 将两个集合的对称差集作为新的集合返回
a.union(b)                                   # 将两个集合的并集作为新的集合返回
c = frozenset()                              # 返回只读集合
```

### 字符串

```python
a = 'red'                      # 赋值
char = a[2]                    # 访问个别字符
'red ' + 'blue'                # 字符串连接
'1, 2, three'.split(',')       # 将字符串拆分成列表
'.'.join(['1', '2', 'three'])  # 将列表连接起来并形成字符串
```

### 操作符

```python
a = 2             # 赋值
a += 1 (*=, /=)   # 更改并赋值
3 + 2             # 加法
3 / 2             # 整数 (python2)或浮点(python3)除法
3 // 2            # 整数除法
3 * 2             # 乘法
3 ** 2            # 指数
3 % 2             # 余数
abs(a)            # 绝对值
1 == 1            # 等于
2 > 1             # 大于
2 < 1             # 小于
1 != 2            # 不等于
1 != 2 and 2 < 3  # 逻辑与
1 != 2 or 2 < 3   # 逻辑或
not 1 == 2        # 逻辑否
'a' in b          # 测试a是否在b中
a is b            # 测试对象是否指向相同的内存区域(id)
```

### 控制流

```python
# if/elif/else条件判断
a, b = 1, 2
if a + b == 3:
    print('True')
elif a + b == 1:
    print('False')
else:
    print('?')

# for循环
a = ['red', 'blue', 'green']
for color in a:
    print(color)

# while循环
number = 1
while number < 10:
    print(number)
    number += 1

# break（跳出循环或条件判断）
number = 1
while True:
    print(number)
    number += 1
    if number > 10:
        break

# continue（跳过本次循环）
for i in range(20):
    if i % 2 == 0:
        continue
    print(i)
```

### 函数, 类, 生成器, 修饰器

```python
# 函数（functions）对代码语句进行分组并可能有返回值
def myfunc(a1, a2):
    return a1 + a2

x = myfunc(a1, a2)

# 类（classes）对属性（数据）和相关方法（函数）进行分组
class Point(object):
    def __init__(self, x):
        self.x = x
    def __call__(self):
        print(self.x)

x = Point(3)

# 生成器（generators）能进行迭代但是不会一次生成所有的值
def firstn(n):
    num = 0
    while num < n:
        yield num
        num += 1

x = [i for i in firstn(10)]

# 修饰器（decorators）可以用来改变函数的行为
class myDecorator(object):
    def __init__(self, f):
        self.f = f
    def __call__(self):
        print("call")
        self.f()

@myDecorator
def my_funct():
    print('func')

my_funct()
```

## NumPy (`import numpy as np`)

### 数组初始化

```python
np.array([2, 3, 4])             # direct initialization
np.empty(20, dtype=np.float32)  # single precision array of size 20
np.zeros(200)                   # initialize 200 zeros
np.ones((3,3), dtype=np.int32)  # 3 x 3 integer matrix with ones
np.eye(200)                     # ones on the diagonal
np.zeros_like(a)                # array with zeros and the shape of a
np.linspace(0., 10., 100)       # 100 points from 0 to 10
np.arange(0, 100, 2)            # points from 0 to <100 with step 2
np.logspace(-5, 2, 100)         # 100 log-spaced from 1e-5 -> 1e2
np.copy(a)                      # copy array to new memory
```

### 索引

```python
a = np.arange(100)          # initialization with 0 - 99
a[:3] = 0                   # set the first three indices to zero
a[2:5] = 1                  # set indices 2-4 to 1
a[:-3] = 2                  # set all but last three elements to 2
a[start:stop:step]          # general form of indexing/slicing
a[None, :]                  # transform to column vector
a[[1, 1, 3, 8]]             # return array with values of the indices
a = a.reshape(10, 10)       # transform to 10 x 10 matrix
a.T                         # return transposed view
b = np.transpose(a, (1, 0)) # transpose array to new axis order
a[a < 2]                    # values with elementwise condition
```

### 数组属性和操作

```python
a.shape                # a tuple with the lengths of each axis
len(a)                 # length of axis 0
a.ndim                 # number of dimensions (axes)
a.sort(axis=1)         # sort array along axis
a.flatten()            # collapse array to one dimension
a.conj()               # return complex conjugate
a.astype(np.int16)     # cast to integer
a.tolist()             # convert (possibly multidimensional) array to list
np.argmax(a, axis=1)   # return index of maximum along a given axis
np.cumsum(a)           # return cumulative sum
np.any(a)              # True if any element is True
np.all(a)              # True if all elements are True
np.argsort(a, axis=1)  # return sorted index array along axis
np.where(cond)         # return indices where cond is True
np.where(cond, x, y)   # return elements from x or y depending on cond
```

### 布尔数组

```python
a < 2                         # returns array with boolean values
(a < 2) & (b > 10)            # elementwise logical and
(a < 2) | (b > 10)            # elementwise logical or
~a                            # invert boolean array
```

### 元素逐个计算操作和数学函数

```python
a * 5              # multiplication with scalar
a + 5              # addition with scalar
a + b              # addition with array b
a / b              # division with b (np.NaN for division by zero)
np.exp(a)          # exponential (complex and real)
np.power(a, b)     # a to the power b
np.sin(a)          # sine
np.cos(a)          # cosine
np.arctan2(a, b)   # arctan(a/b)
np.arcsin(a)       # arcsin
np.radians(a)      # degrees to radians
np.degrees(a)      # radians to degrees
np.var(a)          # variance of array
np.std(a, axis=1)  # standard deviation
```

### 内/外积

```python
np.dot(a, b)                  # inner product: a_mi b_in
np.einsum('ij,kj->ik', a, b)  # einstein summation convention
np.sum(a, axis=1)             # sum over axis 1
np.abs(a)                     # return absolute values
a[None, :] + b[:, None]       # outer sum
a[None, :] * b[:, None]       # outer product
np.outer(a, b)                # outer product
np.sum(a * a.T)               # matrix norm
```


### 线性代数/矩阵数学

```python
evals, evecs = np.linalg.eig(a)      # Find eigenvalues and eigenvectors
evals, evecs = np.linalg.eigh(a)     # np.linalg.eig for hermitian matrix
```


### 读/写文件

```python

np.loadtxt(fname/fobject, skiprows=2, delimiter=',')   # ascii data from file
np.savetxt(fname/fobject, array, fmt='%.5f')           # write ascii data
np.fromfile(fname/fobject, dtype=np.float32, count=5)  # binary data from file
np.tofile(fname/fobject)                               # write (C) binary data
np.save(fname/fobject, array)                          # save as numpy binary (.npy)
np.load(fname/fobject, mmap_mode='c')                  # load .npy file (memory mapped)
```

### 插值，积分，优化

```python
np.trapz(a, x=x, axis=1)  # integrate along axis 1
np.interp(x, xp, yp)      # interpolate function xp, yp at points x
np.linalg.lstsq(a, b)     # solve a x = b in least square sense
```

### 傅立叶变换

```python
np.fft.fft(a)                # complex fourier transform of a
f = np.fft.fftfreq(len(a))   # fft frequencies
np.fft.fftshift(f)           # shifts zero frequency to the middle
np.fft.rfft(a)               # real fourier transform of a
np.fft.rfftfreq(len(a))      # real fft frequencies
```

### 取整

```python
np.ceil(a)   # rounds to nearest upper int
np.floor(a)  # rounds to nearest lower int
np.round(a)  # rounds to neares int
```

### 随机变量

```python
from np.random import normal, seed, rand, uniform, randint
normal(loc=0, scale=2, size=100)  # 100 normal distributed
seed(23032)                       # resets the seed value
rand(200)                         # 200 random numbers in [0, 1)
uniform(1, 30, 200)               # 200 random numbers in [1, 30)
randint(1, 16, 300)               # 300 random integers in [1, 16)
```

## Matplotlib (`import matplotlib.pyplot as plt`)

### 图和坐标轴

```python
fig = plt.figure(figsize=(5, 2))  # initialize figure
fig.savefig('out.png')            # save png image
fig, axes = plt.subplots(5, 2, figsize=(5, 5)) # fig and 5 x 2 nparray of axes
ax = fig.add_subplot(3, 2, 2)     # add second subplot in a 3 x 2 grid
ax = plt.subplot2grid((2, 2), (0, 0), colspan=2)  # multi column/row axis
ax = fig.add_axes([left, bottom, width, height])  # add custom axis
```

### 图和坐标轴属性

```python
fig.suptitle('title')            # big figure title
fig.subplots_adjust(bottom=0.1, right=0.8, top=0.9, wspace=0.2,
                    hspace=0.5)  # adjust subplot positions
fig.tight_layout(pad=0.1, h_pad=0.5, w_pad=0.5,
                 rect=None)      # adjust subplots to fit into fig
ax.set_xlabel('xbla')            # set xlabel
ax.set_ylabel('ybla')            # set ylabel
ax.set_xlim(1, 2)                # sets x limits
ax.set_ylim(3, 4)                # sets y limits
ax.set_title('blabla')           # sets the axis title
ax.set(xlabel='bla')             # set multiple parameters at once
ax.legend(loc='upper center')    # activate legend
ax.grid(True, which='both')      # activate grid
bbox = ax.get_position()         # returns the axes bounding box
bbox.x0 + bbox.width             # bounding box parameters
```

### 画图函数

```python
ax.plot(x,y, '-o', c='red', lw=2, label='bla')  # plots a line
ax.scatter(x,y, s=20, c=color)                  # scatter plot
ax.pcolormesh(xx, yy, zz, shading='gouraud')    # fast colormesh
ax.colormesh(xx, yy, zz, norm=norm)             # slower colormesh
ax.contour(xx, yy, zz, cmap='jet')              # contour lines
ax.contourf(xx, yy, zz, vmin=2, vmax=4)         # filled contours
n, bins, patch = ax.hist(x, 50)                 # histogram
ax.imshow(matrix, origin='lower',
          extent=(x1, x2, y1, y2))              # show image
ax.specgram(y, FS=0.1, noverlap=128,
            scale='linear')                     # plot a spectrogram
ax.text(x, y, string, fontsize=12, color='m')   # write text
```

## Scipy (`import scipy as sci`)

### 插值

```python
# interpolate data at index positions:
from scipy.ndimage import map_coordinates
pts_new = map_coordinates(data, float_indices, order=3)

# simple 1d interpolator with axis argument:
from scipy.interpolate import interp1d
interpolator = interp1d(x, y, axis=2, fill_value=0., bounds_error=False)
y_new = interpolator(x_new)
```

### 积分

```python
from scipy.integrate import quad     # definite integral of python
value = quad(func, low_lim, up_lim)  # function/method
```

### 线性代数

```python
from scipy import linalg
evals, evecs = linalg.eig(a)      # Find eigenvalues and eigenvectors
evals, evecs = linalg.eigh(a)     # linalg.eig for hermitian matrix
b = linalg.expm(a)                # Matrix exponential
c = linalg.logm(a)                # Matrix logarithm
```


## Pandas (`import pandas as pd`)

### 数据结构
```python
s = pd.Series(np.random.rand(1000), index=range(1000))  # series
index = pd.date_range("13/06/2016", periods=1000)       # time index
df = pd.DataFrame(np.zeros((1000, 3)), index=index,
                    columns=["A", "B", "C"])            # DataFrame
```

### 数据框
```python
df = pd.read_csv("filename.csv")   # read and load CSV file in a DataFrame
raw = df.values                    # get raw data out of DataFrame object
cols = df.columns                  # get list of columns headers
df.dtypes                          # get data types of all columns
df.head(5)                         # get first 5 rows
df.describe()                      # get basic statisitics for all columns
df.index                           # get index column range

#column slicing
# (.loc[] and .ix[] are inclusive of the range of values selected)
df.col_name                         # select column values as a series by column name (not optimized)
df[['col_name']]                    # select column values as a dataframe by column name (not optimized)
df.loc[:, 'col_name']               # select column values as a series by column name
df.loc[:, ['col_name']]             # select column values as a dataframe by column name 
df.iloc[:, 0]                       # select by column index
df.iloc[:, [0]]                     # select by column index, but as a dataframe
df.ix[:, 'col_name']                # hybrid approach with column name
df.ix[:, 0]                         # hybrid approach with column index

# row slicing
print(df[:2])                      # print first 2 rows of the dataframe
df.iloc[0:2, :]                    # select first 2 rows of the dataframe
df.loc[0:2,'col_name']             # select first 3 rows of the dataframe
df.loc[0:2, ['col_name1', 'col_name3', 'col_name6']]    # select first 3 rows of the 3 different columns
df.iloc[0:2,0:2]                   # select fisrt 3 rows and first 3 columns
# Again, .loc[] and .ix[] are inclusive

# Dicing
df[ df.col_name < 7 ]                            # select all rows where col_name < 7
df[ (df.col_name1 < 7) & (df.col_name2 == 0) ]       # combine multiple boolean indexing conditionals using bit-wise logical operators.
                                                     # Regular Python boolean operators (and, or) cannot be used here. 
                                                     # Be sure to encapsulate each conditional in parenthesis to make this work.
df[df.recency < 7] = -100                        # writing to slice
```