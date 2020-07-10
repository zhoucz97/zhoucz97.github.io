---
layout: post
title: 莫烦numpy+pandas教程笔记
date: 2020-07-09
categories: 学习笔记
tags: python numpy pandas
---

> 莫烦小哥哥的教程，很简短，numpy和pandas一起的，一共大概三个小时，很快就可以看完，入门佳选。

### [视频地址](https://www.bilibili.com/video/BV1Ex411L7oT?from=search&seid=8015826183781802668)

### 我的笔记，用jupyter写的

网盘里是.md和.ipynb格式  
链接：https://pan.baidu.com/s/1Y2EZda03sHHhdo7WqY5mCA   
提取码：bc6h  

### 笔记内容

# 1.numpy笔记

## numpy array

1. `np.array(list)`创建一个array  
`np.array(list, dtype=np.int)` dtype可以指定数据类型  
`np.zeros(shape)` 创建全0的array  
`np.ones(shape)` 创建全1的array  
`np.empty(shape)` 创建随机array  
`np.arange([start], end, [step], [dtype])`  
`np.linspace(start, end, sections)`从start到end分成若干段
2. `array.ndim`array的维数
3. `array.shape`array的形状
4. `array.size`array的元素个数



```python
import numpy as np
```


```python
arr = np.array([[1,2,3], [2,3,4]])
```


```python
print(arr)
np.linspace()
print(arr.ndim)
print(arr.shape)
print(arr.size)
```

    [[1 2 3]
     [2 3 4]]
    2
    (2, 3)
    6
    

## numpy基础运算

- `array1 + array2` array对应位相加
- `array1 * array2` 对应位相乘
- `array1.dot(array2)`=`numpy.dot(array1, array2)` 矩阵乘法
- `numpy.random.random(shape)` 生成shape形状的0~1的随机array
- `np.sum(array, axis)` 按某维度求和
- `np.min(array, axis)` 按某维度求最小值
- `np.max(array, axis)` 按某维度求最大值
- #### 关于axis!!  二维的话0表示列，1表示行；多维的话记住，0就是第0层，也就是最外层，1是第1层，n是第n层，也就是最内层
- `np.argmin(A)` array的最小值所对应的索引值
- `np.mean(array)` =  `array.mean()` = `np.average(array)`array的平均值
- `np.median(array)` array的中位数
- `np.cumsum(array)` 逐位累加，输出和array形状一样的数组，见样例
- `np.diff(array)` array中一个数和前一个数之差，见样例
- `np.nonzero(array)` 输出两个array,分别是非零元素的行和列
- `np.sort(array)` 排序  axis=1按行排序（默认），axis=0按列排序
- `array.T` = `np.transpose(array)` array的转置
- `np.clip(array, min, max)` 将array中小于min的值都设成min,大于max的值都设成max



```python
import numpy as np
```


```python
a = np.array([10,20,30,40])
b = np.arange(4)
print(a , b)
```

    [10 20 30 40] [0 1 2 3]
    


```python
print(a + b)
```

    [10 21 32 43]
    


```python
b**2
```




    array([0, 1, 4, 9], dtype=int32)




```python
b<3
```




    array([ True,  True,  True, False])




```python
a = np.array([[1,2], [3,4]])
b = np.arange(4).reshape((2,2))
```


```python
c = a * b
print(c)
```

    [[ 0  2]
     [ 6 12]]
    


```python
d = a.dot(b)
print(d)
print(np.dot(a, b))
```

    [[ 4  7]
     [ 8 15]]
    [[ 4  7]
     [ 8 15]]
    


```python
np.min(a, axis=0)
```




    array([1, 2])




```python
np.cumsum(a)
```




    array([ 1,  3,  6, 10], dtype=int32)




```python
np.diff([[1,2,3,4], [5,6,10,8]])
```




    array([[ 1,  1,  1],
           [ 1,  4, -2]])




```python
arr = np.random.random((2,3,4))
arr
```




    array([[[0.21008032, 0.1192331 , 0.00914905, 0.72947293],
            [0.02839615, 0.09864839, 0.39741167, 0.57597824],
            [0.96836791, 0.51862262, 0.88087796, 0.46328493]],
    
           [[0.65260805, 0.53992939, 0.56990468, 0.24925595],
            [0.22224978, 0.30332973, 0.09053094, 0.52629985],
            [0.68196468, 0.58885901, 0.27143766, 0.57624114]]])




```python
np.sort(arr, axis=0)
```




    array([[[0.21008032, 0.1192331 , 0.00914905, 0.24925595],
            [0.02839615, 0.09864839, 0.09053094, 0.52629985],
            [0.68196468, 0.51862262, 0.27143766, 0.46328493]],
    
           [[0.65260805, 0.53992939, 0.56990468, 0.72947293],
            [0.22224978, 0.30332973, 0.39741167, 0.57597824],
            [0.96836791, 0.58885901, 0.88087796, 0.57624114]]])




```python
np.clip(a, 2, 3)
```




    array([[2, 2],
           [3, 3]])



## numpy索引
用法和list差不多，支持切片


```python
import numpy as np
```


```python
A = np.arange(3, 15).reshape((3, 4))
```


```python
print(A, '\n')
print(A[2], '\n')
print(A[2, :], '\n')
print(A[2, 0: 2], '\n')
print(A[2][0], '\n')
print(A[2, 0])
```

    [[ 3  4  5  6]
     [ 7  8  9 10]
     [11 12 13 14]] 
    
    [11 12 13 14] 
    
    [11 12 13 14] 
    
    [11 12] 
    
    11 
    
    11
    


```python
for row in A:
    print(row)
for column in A.T:   #可以按列遍历
    print(column)
```

    [3 4 5 6]
    [ 7  8  9 10]
    [11 12 13 14]
    [ 3  7 11]
    [ 4  8 12]
    [ 5  9 13]
    [ 6 10 14]
    

#### 1.A.flatten()  将A变为一维数组输出
#### 2.for item in A.flat  逐个输出A的值


```python
print(A.flatten())
for item in A.flat:
    print(item, end=' ')
```

    [ 3  4  5  6  7  8  9 10 11 12 13 14]
    3 4 5 6 7 8 9 10 11 12 13 14 

## array合并

1. `array3 = np.vstack((array1, array2))`vertical stack 垂直叠加，两个数组的维度可以不同
2. `array3 = np.hstack((array1, array2))`horizontal stack 水平叠加，要求两个数组的维度必须相同
3. `array3 = np.concatenate((A,B,B,A), axis=0)`axis=0在列这个维度进行合并，axis=1在行这个维度进行合并


```python
A = np.array([[1,1,1],[3,3,3]])
B = np.array([[2,2,2], [4,4,4]])
```


```python
C = np.vstack((A, B))   # vertical stack 垂直叠加
print(A.shape)
print(C.shape)
print(C)
E = np.concatenate((A, B), axis=0)
print(E)
```

    (2, 3)
    (4, 3)
    [[1 1 1]
     [3 3 3]
     [2 2 2]
     [4 4 4]]
    [[1 1 1]
     [3 3 3]
     [2 2 2]
     [4 4 4]]
    


```python
D = np.hstack((A, B))
print(D)
```

    [[1 1 1 2 2 2]
     [3 3 3 4 4 4]]
    

- `np.newaxis`在指定位置添加一个维度


```python
print(A)
```

    [1 1 1]
    


```python
print(A[:, np.newaxis])
print(A[np.newaxis, :])
```

    [[1]
     [1]
     [1]]
    [[1 1 1]]
    

## array分割

1. `np.split(ary, indices_or_sections, axis)`均等分割，把ary分割成若干块，按axis进行分割
2. `np.array_split()` 不均等分割，第0项分割出来的元素最多，其余均等分 
3. `np.vsplit(A, section)`纵向分割，分成section块，相当于axis=0
4. `np.hsplit(A, section)`横向分割，分成section块，相当于axis=1

[np.split()与np.array_split()函数](https://www.cnblogs.com/shona/p/12163515.html)


```python
A = np.arange(12).reshape(3,4)
A
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])




```python
a1, a2 = np.split(A, 2, 1)
print(a1, '\n\n', a2)
```

    [[0 1]
     [4 5]
     [8 9]] 
    
     [[ 2  3]
     [ 6  7]
     [10 11]]
    


```python
print(np.vsplit(A, 1))
```

    [array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])]
    


```python
print(np.hsplit(A, 2))
```

    [array([[0, 1],
           [4, 5],
           [8, 9]]), array([[ 2,  3],
           [ 6,  7],
           [10, 11]])]
    

## numpy copy

1. `b = a` 和`b = a[:]`为浅copy, `b is a`为`True`.   
注：python 的列表中`b = a[:]`为深copy
2. `b = a.copy()`为深copy


```python
a = np.arange(4)
a
```




    array([0, 1, 2, 3])




```python
b = a
b
```




    array([0, 1, 2, 3])




```python
a[0] = -1
b
```




    array([-1,  1,  2,  3])




```python
c = a.copy()
print(c)
a[0] = 1122
print(c)

```

    [-1  1  2  3]
    [-1  1  2  3]
    


```python
d = a[:]
print(d)
a[0] = 11111
print(d)
```

    [1122    1    2    3]
    [11111     1     2     3]
    


# 2.pandas笔记

## 一.Pandas基本介绍

pandas像是字典形式的numpy

1. `pd.Series()` 可以定义一个带有轴标签的一位ndarray，Series里面可以放多种类型的数据
2. `pd.date_range(start, end, periods)` 生成DatetimeIndex类型的时间序列
3. `pd.DataFrame(data, index, columns, dtype)` 生成二维表格数据，可包含多种类型的数据。index表示行名，columns表示列名
4. `df.index  df.columns  df.dtypes  df.values()`分别为DataFrame数据的行名，列名，数据类型，按array形式输出df的内容
5. `df.describe()`  输出df中为数字类型数据的各种性质，如平均数中位数等等
6. `df.T`  输出转置


```python
import pandas as pd
import numpy as np
```


```python
s = pd.Series([1, 3, 6, np.nan, 44, 1])
s
pd.Series()
```




    0     1.0
    1     3.0
    2     6.0
    3     NaN
    4    44.0
    5     1.0
    dtype: float64




```python
dates = pd.date_range(start='20160101', end='20160108', periods=6)
dates
```




    DatetimeIndex(['2016-01-01 00:00:00', '2016-01-02 09:36:00',
                   '2016-01-03 19:12:00', '2016-01-05 04:48:00',
                   '2016-01-06 14:24:00', '2016-01-08 00:00:00'],
                  dtype='datetime64[ns]', freq=None)




```python
# index是行的名称，columns是列的名称
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=['a','b','c','d'])
df
print(type(df))
```

    <class 'pandas.core.frame.DataFrame'>
    


```python
df2 = pd.DataFrame({
    'A': 1. ,
    'B': pd.Timestamp('20200709') ,
    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
    'E': pd.Categorical(['test', 'train', 'test', 'train']),
    'F': 'foo',
})
df2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>E</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2020-07-09</td>
      <td>1.0</td>
      <td>test</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>2020-07-09</td>
      <td>1.0</td>
      <td>train</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>2020-07-09</td>
      <td>1.0</td>
      <td>test</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>2020-07-09</td>
      <td>1.0</td>
      <td>train</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2.dtypes
```




    A           float64
    B    datetime64[ns]
    C           float32
    E          category
    F            object
    dtype: object




```python
df2.index
```




    Int64Index([0, 1, 2, 3], dtype='int64')




```python
df2.columns
```




    Index(['A', 'B', 'C', 'E', 'F'], dtype='object')




```python
df2.values  # 输出df2的内容
```




    array([[1.0, Timestamp('2020-07-09 00:00:00'), 1.0, 'test', 'foo'],
           [1.0, Timestamp('2020-07-09 00:00:00'), 1.0, 'train', 'foo'],
           [1.0, Timestamp('2020-07-09 00:00:00'), 1.0, 'test', 'foo'],
           [1.0, Timestamp('2020-07-09 00:00:00'), 1.0, 'train', 'foo']],
          dtype=object)




```python
df2.describe()  # 计算数字形式的各种属性
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2.T  #输出转置，也就是行列互换
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>B</th>
      <td>2020-07-09 00:00:00</td>
      <td>2020-07-09 00:00:00</td>
      <td>2020-07-09 00:00:00</td>
      <td>2020-07-09 00:00:00</td>
    </tr>
    <tr>
      <th>C</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>E</th>
      <td>test</td>
      <td>train</td>
      <td>test</td>
      <td>train</td>
    </tr>
    <tr>
      <th>F</th>
      <td>foo</td>
      <td>foo</td>
      <td>foo</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2.sort_index(axis=0, ascending=False) # axis=0表示按index进行排序，=1表示按columns进行排序
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>E</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>2020-07-09</td>
      <td>1.0</td>
      <td>train</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>2020-07-09</td>
      <td>1.0</td>
      <td>test</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>2020-07-09</td>
      <td>1.0</td>
      <td>train</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2020-07-09</td>
      <td>1.0</td>
      <td>test</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2.sort_values(by='E')  # by表示对谁进行排序
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>E</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2020-07-09</td>
      <td>1.0</td>
      <td>test</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>2020-07-09</td>
      <td>1.0</td>
      <td>test</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>2020-07-09</td>
      <td>1.0</td>
      <td>train</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>2020-07-09</td>
      <td>1.0</td>
      <td>train</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>



## 二、选择数据

- `df['A']` = `df.A` 选择某一列
- `df[['A', 'B']]` 选择若干列
- `df[row1 : row2]` 切片选择的是行
- `df[row1_name : row2_name]` 还可以用行名来选择行
- `df.loc[rowname, columnname]` 按照标签名进行选择，标签名可以是行名，也可以是列名。支持切片，也支持列表  
注意：df.loc是双闭区间，不是Python传统的左闭右开
- `df.iloc[rowindex, columnindex]`按照位置进行选择

[Pandas中loc，iloc与直接切片的区别](https://www.cnblogs.com/daozhongshu/archive/2018/04/30/8973439.html)


```python
import pandas as pd
import numpy as np
```


```python
dates = pd.date_range('20200710', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A','B','C','D'])
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-07-10</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2020-07-11</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2020-07-12</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2020-07-13</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
    <tr>
      <th>2020-07-14</th>
      <td>16</td>
      <td>17</td>
      <td>18</td>
      <td>19</td>
    </tr>
    <tr>
      <th>2020-07-15</th>
      <td>20</td>
      <td>21</td>
      <td>22</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df['A'], '\n', df.A)
```

    2020-07-10     0
    2020-07-11     4
    2020-07-12     8
    2020-07-13    12
    2020-07-14    16
    2020-07-15    20
    Freq: D, Name: A, dtype: int32 
     2020-07-10     0
    2020-07-11     4
    2020-07-12     8
    2020-07-13    12
    2020-07-14    16
    2020-07-15    20
    Freq: D, Name: A, dtype: int32
    


```python
print(df[:3])
```

                A  B   C   D
    2020-07-10  0  1   2   3
    2020-07-11  4  5   6   7
    2020-07-12  8  9  10  11
    


```python
print(df['20200710' : '20200713'])
```

                 A   B   C   D
    2020-07-10   0   1   2   3
    2020-07-11   4   5   6   7
    2020-07-12   8   9  10  11
    2020-07-13  12  13  14  15
    


```python
print(df.loc['20200710':'20200712', 'A' : 'D'])
df.loc()
```

                A  B   C   D
    2020-07-10  0  1   2   3
    2020-07-11  4  5   6   7
    2020-07-12  8  9  10  11
    




    <pandas.core.indexing._LocIndexer at 0x2547d798318>




```python
print(df.iloc[3, 1:3])
```

    B    13
    C    14
    Name: 2020-07-13 00:00:00, dtype: int32
    


```python
print(df.loc['20200710', 'A'])
```

    0
    


```python
print(df[df.A > 8])
```

                 A   B   C   D
    2020-07-13  12  13  14  15
    2020-07-14  16  17  18  19
    2020-07-15  20  21  22  23
    

## 三、设置值

利用`df.loc`、`df.iloc`进行设置


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-07-10</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2020-07-11</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2020-07-12</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2020-07-13</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
    <tr>
      <th>2020-07-14</th>
      <td>16</td>
      <td>17</td>
      <td>18</td>
      <td>19</td>
    </tr>
    <tr>
      <th>2020-07-15</th>
      <td>20</td>
      <td>21</td>
      <td>22</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iloc[2,2] = 111
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-07-10</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2020-07-11</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2020-07-12</th>
      <td>8</td>
      <td>9</td>
      <td>111</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2020-07-13</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
    <tr>
      <th>2020-07-14</th>
      <td>16</td>
      <td>17</td>
      <td>18</td>
      <td>19</td>
    </tr>
    <tr>
      <th>2020-07-15</th>
      <td>20</td>
      <td>21</td>
      <td>22</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc['20200710', 'D'] = 999
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-07-10</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>999</td>
    </tr>
    <tr>
      <th>2020-07-11</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2020-07-12</th>
      <td>8</td>
      <td>9</td>
      <td>111</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2020-07-13</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
    <tr>
      <th>2020-07-14</th>
      <td>16</td>
      <td>17</td>
      <td>18</td>
      <td>19</td>
    </tr>
    <tr>
      <th>2020-07-15</th>
      <td>20</td>
      <td>21</td>
      <td>22</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df.A > 4] = 0
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-07-10</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>999</td>
    </tr>
    <tr>
      <th>2020-07-11</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2020-07-12</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2020-07-13</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2020-07-14</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2020-07-15</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.A[df.A>=4] = 99999
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-07-10</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>999</td>
    </tr>
    <tr>
      <th>2020-07-11</th>
      <td>99999</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2020-07-12</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2020-07-13</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2020-07-14</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2020-07-15</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['F'] = np.nan  # 加一列
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-07-10</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>999</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2020-07-11</th>
      <td>99999</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2020-07-12</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2020-07-13</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2020-07-14</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2020-07-15</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['E'] = pd.Series(np.arange(6), index=df.index)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>F</th>
      <th>E</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-07-10</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>999</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2020-07-11</th>
      <td>99999</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2020-07-12</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2020-07-13</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2020-07-14</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2020-07-15</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



## 4.处理丢失数据

1. `df.dropna(axis, how)`   
axis=0表示丢掉行，axis=1表示丢掉列；  
how='any'表示任意一个是nan就丢掉整行或整列,how='all'表示整行或整列都是nan才丢掉
2. `df.fillna(value)` 用value填充缺失值
3. `df.isna()` = `df.isnull()` 判断是否为缺失值


```python
import pandas as pd
import numpy as np
```


```python
dates = pd.date_range('20200710', periods=6)
```


```python
df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', 'C', 'D'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-07-10</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2020-07-11</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2020-07-12</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2020-07-13</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
    <tr>
      <th>2020-07-14</th>
      <td>16</td>
      <td>17</td>
      <td>18</td>
      <td>19</td>
    </tr>
    <tr>
      <th>2020-07-15</th>
      <td>20</td>
      <td>21</td>
      <td>22</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iloc[0, 1] = np.nan
df.iloc[1, 2] = np.nan
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-07-10</th>
      <td>0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2020-07-11</th>
      <td>4</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2020-07-12</th>
      <td>8</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2020-07-13</th>
      <td>12</td>
      <td>13.0</td>
      <td>14.0</td>
      <td>15</td>
    </tr>
    <tr>
      <th>2020-07-14</th>
      <td>16</td>
      <td>17.0</td>
      <td>18.0</td>
      <td>19</td>
    </tr>
    <tr>
      <th>2020-07-15</th>
      <td>20</td>
      <td>21.0</td>
      <td>22.0</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.dropna(axis=1, how='any')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-07-10</th>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2020-07-11</th>
      <td>4</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2020-07-12</th>
      <td>8</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2020-07-13</th>
      <td>12</td>
      <td>15</td>
    </tr>
    <tr>
      <th>2020-07-14</th>
      <td>16</td>
      <td>19</td>
    </tr>
    <tr>
      <th>2020-07-15</th>
      <td>20</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.fillna(value=0)
df.fillna()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-07-10</th>
      <td>0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2020-07-11</th>
      <td>4</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2020-07-12</th>
      <td>8</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2020-07-13</th>
      <td>12</td>
      <td>13.0</td>
      <td>14.0</td>
      <td>15</td>
    </tr>
    <tr>
      <th>2020-07-14</th>
      <td>16</td>
      <td>17.0</td>
      <td>18.0</td>
      <td>19</td>
    </tr>
    <tr>
      <th>2020-07-15</th>
      <td>20</td>
      <td>21.0</td>
      <td>22.0</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.isna()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-07-10</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2020-07-11</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2020-07-12</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2020-07-13</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2020-07-14</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2020-07-15</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.isnull()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-07-10</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2020-07-11</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2020-07-12</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2020-07-13</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2020-07-14</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2020-07-15</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



## 5.导入导出数据

- `data = pd.read_csv('student.csv')` 导入数据
- `data.to_pickle('student.pickle')`  导出数据

## 6.合并concat

1. `pd.concat([df1, df2, ...], axis, ignore_index, join)`   
参数说明：
    - axis=0竖向合并，axis=1横向合并；  
    - ignore_index=True忽略索引名，全部改为01234...  
    - join默认为'outer'，也就是数据库中的全连接，‘inner’为内连接
2. `df1.append([df1, df2, ..], ignore_index)` 类似于concat的join='outer'且axis=0的合并


```python
df1 = pd.DataFrame(np.ones((3, 4))*0, columns=['a','b','c','d'])
df1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2 = pd.DataFrame(np.ones((3, 4))*1, columns=['a','b','c','d'])
df2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df3 = pd.DataFrame(np.ones((3, 4))*2, columns=['a','b','c','d'])
df3
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
res = pd.concat([df1, df2, df3],axis=0, ignore_index=True)
res
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1 = pd.DataFrame(np.ones((3, 4))*0, columns=['a','b','c','d'], index=[1,2,3])
df1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2 = pd.DataFrame(np.ones((3, 4))*1, columns=['b','c','d','e'], index=[2,3, 4])
df2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
res = df1.append([df2, df2], ignore_index=True)
res
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
s1 = pd.Series([1,2,3,4], index=['a','b','c','d'])
s1
```




    a    1
    b    2
    c    3
    d    4
    dtype: int64




```python
res = df1.append(s1, ignore_index=True)
res
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>



## 7.合并Merge

1. `pd.merge(df1, df2, on, how, indicator)` 将df1和df2按照on指定的索引值相同的进行合并  
    - how可取left,right,outer,inner，分别为左外连接，右外连接，外连接，内连接。默认为inner
    - indicator显示连接方式，可自定义名字
    - df1_index=True, df2_index=True表示按照行名进行合并，（默认是按列名进行合并的）
    - suffixes 可以区分同名列


```python
left = pd.DataFrame({'key':['K0','K1','K2','K3'],
                    'A':['A0','A1','A2','A3'],
                    'B':['B0','B1','B2','B3'],
                    })
left
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>K0</td>
      <td>A0</td>
      <td>B0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>K1</td>
      <td>A1</td>
      <td>B1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>K2</td>
      <td>A2</td>
      <td>B2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>K3</td>
      <td>A3</td>
      <td>B3</td>
    </tr>
  </tbody>
</table>
</div>




```python
right = pd.DataFrame({'key':['K0','K1','K2','K3'],
                    'C':['C0','C1','C2','C3'],
                    'D':['D0','D1','D2','D3'],
                    })
right
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>K0</td>
      <td>C0</td>
      <td>D0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>K1</td>
      <td>C1</td>
      <td>D1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>K2</td>
      <td>C2</td>
      <td>D2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>K3</td>
      <td>C3</td>
      <td>D3</td>
    </tr>
  </tbody>
</table>
</div>




```python
res = pd.merge(left, right, on='key')
res
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>K0</td>
      <td>A0</td>
      <td>B0</td>
      <td>C0</td>
      <td>D0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>K1</td>
      <td>A1</td>
      <td>B1</td>
      <td>C1</td>
      <td>D1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>K2</td>
      <td>A2</td>
      <td>B2</td>
      <td>C2</td>
      <td>D2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>K3</td>
      <td>A3</td>
      <td>B3</td>
      <td>C3</td>
      <td>D3</td>
    </tr>
  </tbody>
</table>
</div>




```python
left = pd.DataFrame({'key1':['K0','K0','K1','K2'],
                     'key2':['K0','K1','K0','K1'],
                    'A':['A0','A1','A2','A3'],
                    'B':['B0','B1','B2','B3'],
                    })
left
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key1</th>
      <th>key2</th>
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>K0</td>
      <td>K0</td>
      <td>A0</td>
      <td>B0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>K0</td>
      <td>K1</td>
      <td>A1</td>
      <td>B1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>K1</td>
      <td>K0</td>
      <td>A2</td>
      <td>B2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>K2</td>
      <td>K1</td>
      <td>A3</td>
      <td>B3</td>
    </tr>
  </tbody>
</table>
</div>




```python
right = pd.DataFrame({'key1':['K0','K1','K1','K2'],
                      'key2':['K0','K0','K0','K0'],
                    'C':['C0','C1','C2','C3'],
                    'D':['D0','D1','D2','D3'],
                    })
right
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key1</th>
      <th>key2</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>K0</td>
      <td>K0</td>
      <td>C0</td>
      <td>D0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>K1</td>
      <td>K0</td>
      <td>C1</td>
      <td>D1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>K1</td>
      <td>K0</td>
      <td>C2</td>
      <td>D2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>K2</td>
      <td>K0</td>
      <td>C3</td>
      <td>D3</td>
    </tr>
  </tbody>
</table>
</div>




```python
res = pd.merge(left, right, on=['key1', 'key2'])
res
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key1</th>
      <th>key2</th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>K0</td>
      <td>K0</td>
      <td>A0</td>
      <td>B0</td>
      <td>C0</td>
      <td>D0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>K1</td>
      <td>K0</td>
      <td>A2</td>
      <td>B2</td>
      <td>C1</td>
      <td>D1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>K1</td>
      <td>K0</td>
      <td>A2</td>
      <td>B2</td>
      <td>C2</td>
      <td>D2</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1 = pd.DataFrame({'col1':[0, 1], 'col_left':['a', 'b']})
df2 = pd.DataFrame({'col1':[1, 2, 2], 'col_right':[2,2,2]})
print(df1, '\n', df2)
```

       col1 col_left
    0     0        a
    1     1        b 
        col1  col_right
    0     1          2
    1     2          2
    2     2          2
    


```python
res = pd.merge(df1, df2, on='col1', how='outer', indicator=True)
res
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col1</th>
      <th>col_left</th>
      <th>col_right</th>
      <th>_merge</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>a</td>
      <td>NaN</td>
      <td>left_only</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>b</td>
      <td>2.0</td>
      <td>both</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>right_only</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>right_only</td>
    </tr>
  </tbody>
</table>
</div>




```python
left = pd.DataFrame({
                    'A':['A0','A1','A2'],
                    'B':['B0','B1','B2'],
                    },
    index=['K0','K1','K2'])
left
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>K0</th>
      <td>A0</td>
      <td>B0</td>
    </tr>
    <tr>
      <th>K1</th>
      <td>A1</td>
      <td>B1</td>
    </tr>
    <tr>
      <th>K2</th>
      <td>A2</td>
      <td>B2</td>
    </tr>
  </tbody>
</table>
</div>




```python
right = pd.DataFrame({
                    'C':['C0','C1','C2'],
                    'D':['D0','D1','D2'],
                    },
    index=['K0','K2','K3'])
right
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>K0</th>
      <td>C0</td>
      <td>D0</td>
    </tr>
    <tr>
      <th>K2</th>
      <td>C1</td>
      <td>D1</td>
    </tr>
    <tr>
      <th>K3</th>
      <td>C2</td>
      <td>D2</td>
    </tr>
  </tbody>
</table>
</div>




```python
res = pd.merge(left, right, right_index=True, left_index=True, how='outer')
res

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>K0</th>
      <td>A0</td>
      <td>B0</td>
      <td>C0</td>
      <td>D0</td>
    </tr>
    <tr>
      <th>K1</th>
      <td>A1</td>
      <td>B1</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>K2</th>
      <td>A2</td>
      <td>B2</td>
      <td>C1</td>
      <td>D1</td>
    </tr>
    <tr>
      <th>K3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>C2</td>
      <td>D2</td>
    </tr>
  </tbody>
</table>
</div>




```python
boy = pd.DataFrame({
                    'K':['K0','K1','K2'],
                    'age': [1,2,3]})
boy
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>K</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>K0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>K1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>K2</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
girl = pd.DataFrame({
                    'K':['K0','K0','K3'],
                    'age': [4,5,6]})
girl
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>K</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>K0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>K0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>K3</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
res = pd.merge(boy, girl, on='K', suffixes=['_boy', '_girl'], how='outer')
res
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>K</th>
      <th>age_boy</th>
      <th>age_girl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>K0</td>
      <td>1.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>K0</td>
      <td>1.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>K1</td>
      <td>2.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>K2</td>
      <td>3.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>K3</td>
      <td>NaN</td>
      <td>6.0</td>
    </tr>
  </tbody>
</table>
</div>



## 8.plot画图


```python
import matplotlib.pyplot as plt
```


```python
data = pd.Series(np.random.randn(1000), index=np.arange(1000))
data
```




    0     -0.924241
    1      0.261020
    2      1.024094
    3      1.274399
    4      0.533373
             ...   
    995   -0.530959
    996   -1.092390
    997   -0.355699
    998   -0.288971
    999   -0.464878
    Length: 1000, dtype: float64




```python
data = data.cumsum()
data.plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x26cd30e3ac8>




![png](/images/posts/2020/07/1010.png)



```python
data = pd.DataFrame(np.random.randn(1000, 4), index=np.arange(1000), columns=list('ABCD'))
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.605889</td>
      <td>1.547269</td>
      <td>1.505041</td>
      <td>-1.036288</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.527759</td>
      <td>0.436525</td>
      <td>-1.056520</td>
      <td>-0.464886</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.053846</td>
      <td>-0.180606</td>
      <td>-0.558349</td>
      <td>-0.267518</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.067628</td>
      <td>1.209165</td>
      <td>-1.032632</td>
      <td>-0.464935</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.257883</td>
      <td>-1.667595</td>
      <td>0.559738</td>
      <td>-0.174757</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>-0.662562</td>
      <td>0.660993</td>
      <td>-0.054447</td>
      <td>-0.368030</td>
    </tr>
    <tr>
      <th>996</th>
      <td>-0.399942</td>
      <td>-0.799058</td>
      <td>0.589180</td>
      <td>-1.934952</td>
    </tr>
    <tr>
      <th>997</th>
      <td>-0.581433</td>
      <td>-0.774302</td>
      <td>-0.369005</td>
      <td>1.320763</td>
    </tr>
    <tr>
      <th>998</th>
      <td>0.097296</td>
      <td>-2.510756</td>
      <td>2.287460</td>
      <td>1.534604</td>
    </tr>
    <tr>
      <th>999</th>
      <td>-0.408239</td>
      <td>-0.625151</td>
      <td>-1.895787</td>
      <td>0.612191</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 4 columns</p>
</div>




```python
data = data.cumsum(axis=0)
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.605889</td>
      <td>1.547269</td>
      <td>1.505041</td>
      <td>-1.036288</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.133648</td>
      <td>1.983794</td>
      <td>0.448521</td>
      <td>-1.501175</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5.187494</td>
      <td>1.803188</td>
      <td>-0.109827</td>
      <td>-1.768692</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.255122</td>
      <td>3.012353</td>
      <td>-1.142459</td>
      <td>-2.233627</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.513006</td>
      <td>1.344758</td>
      <td>-0.582721</td>
      <td>-2.408385</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>-65.686659</td>
      <td>19.669692</td>
      <td>11.981639</td>
      <td>-17.186383</td>
    </tr>
    <tr>
      <th>996</th>
      <td>-66.086601</td>
      <td>18.870634</td>
      <td>12.570820</td>
      <td>-19.121335</td>
    </tr>
    <tr>
      <th>997</th>
      <td>-66.668034</td>
      <td>18.096331</td>
      <td>12.201815</td>
      <td>-17.800572</td>
    </tr>
    <tr>
      <th>998</th>
      <td>-66.570738</td>
      <td>15.585575</td>
      <td>14.489275</td>
      <td>-16.265967</td>
    </tr>
    <tr>
      <th>999</th>
      <td>-66.978977</td>
      <td>14.960424</td>
      <td>12.593488</td>
      <td>-15.653776</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 4 columns</p>
</div>




```python
data.plot()
plt.show()
```


![png](/images/posts/2020/07/1011.png)



```python
ax = data.plot.scatter(x='A', y='B', color='DarkBlue', label ='Class 1')
data.plot.scatter(x='A', y='C', color='DarkGreen', label ='Class 2', ax=ax)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x26cda898d68>




![png](/images/posts/2020/07/1012.png)



```python

```


