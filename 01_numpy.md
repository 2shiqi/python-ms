## 数组创建、访问
- ndarray.ndim - 数组的轴（维度）个数，维度的数量被称为rank,   行
- ndarry.shape - 每个维度中数组的大小             列
n行m列的矩阵  shape时(n,m)
shape元组的长度是就是rank或维度的个数的ndim
- ndarry.size 数组元素的总数，shape的元素的乘积
- ndarry.dtype 描述数组中元素类型的对象，可以用标准python类型创建或指定dtype，NumPy也提供自己的类型，例如numpy.int32,numpy.int16和numpy.float64
- ndarry.itemsize 数组中每个元素的字节大小
```python
imoprt numpy as np
data=np.array([1,2,3,4,5,6,7,8,9])

print ('Numpy的一维数组:\n{0}'.format(data))
print ('一维数组中各元素扩大十倍:\n{0}'.format(data*10))


data=np.arrpy([[1,3,5,7,9],[2,4,6,8,10]])
print('访问二维数组中第一行第二列元素:{0}'.format(data[0,1]))
print('访问二维数组中第一行第二至四列元素:{0}'.format(data[0,1:4]))
print('访问二维数组中第一行上的所有元素:{0}'.format(data[0,:]))

```
## 数组计算
```python
Myarray2=np.arrage(10)
print('MyArray2:\n{0}'.format(MyArray2))
print('MyArray2的基本描述统计量:\n均值: %f,标准差:%f,总和:%f,最大值:%f'%(MyArray2.mean(),MyArray2.std(),MyArray2.sum(),MyArray2.max()))
print('MyArray2的累计和:{0}'.format(MyArray2.cumsum()))
print('MyArray2开平方:{0}'.format(np.sqrt(MyArray2)))

np.random.seed(123)
MyArray3=np.random.randn(10)
print('MyArray3:\n{0}'.format(MyArray3))

print('MyArray3排序结果:\n{0}'.format(np.sort(MyArray3)))
print('MyArray3四舍五入到最近整数:\n{0}'.format(np.rint(MyArray3)))

print('MyArray3各元素的正负号:{0}'.format(np.sign(MyArray3)))
print('MyArray3各元素非负数的显示"正"，负数显示"负":\n{0}'.format(np.where(MyArray3>0,'正','负’)))

print('MyArray2+MyArray3的结果:\n{0}'.format(MyArray2+MyArray3))

```
数组的方法 mean()-均值，std()-标准差 
==cumsum()==  计算数组元素的当前累计和
==seed==指定随机数种子。目的是确保每次执行代码时，生成的随机数可以再现,否则，每次运行代码，生成的随机数会不相同
==np.random.randn(10)== 生成包含10个元素且服从标准正态分布的1维数组。
==sort== 对数组元素排序，排序结果并不覆盖原数组内容
==rint()== 对数组元素做四舍五入。 
==sign()== 求数组元素的正负符号。1表示正号，-1表示负号
==where()== 依次对数组元素进行逻辑判读。 where()需指定判断条件（如>0），满足条件的返回第一个值（如‘正’），否则返回第二个值（如‘负’）。若省略第2和第3个参数，例如：where(Myarray3>0)将给出满足条件的元素索引号。 
==MyArray2+MyArray3== 将两个数组相同位置上的元素相加


## 矩阵创建与乘法

```python

np.random.seed(123)
X=np.floor(np.random.normal(5,1,(2,5)))
Y=np.eye(5)
print('X:\n{0}'.format(X))
print('Y:\n{0}'.format(Y))
print('X和Y的矩阵积：\n{0}'.format(np.dot(X,Y)))

```
==random.normal()== 生成2行5列的2维数组，数组元素服从均值为5，标准差为1的正态分布。
==floor()== 函数得到距各数组元素最近的最大整数,np.floor(-0.3)==-1.0
==eye()== 生成一个大小（这里是5行5列）的单位阵Y
==dot()== 计算矩阵X和矩阵Y（单位阵）的矩阵乘积，将得到2行5列的矩阵。


## 矩阵运算
```python
from numpy.linalg import inv,svd,eig,det
X=np.random.randn(5,5)
print(X)
mat=X.T.dot(X)   # 点乘: X.T * X
print(mat)
print('矩阵mat的逆：\n{0}'.format(inv(mat)))
print('矩阵mat的行列式值：\n{0}'.format(det(mat)))
print('矩阵mat的特征值和特征向量：\n{0}'.format(eig(mat)))
print('对矩阵mat做奇异值分解：\n{0}'.format(svd(mat)))

```

==np.random.randn(5,5)== 生成5行5列的2维数组X（可视为一个矩阵），数组元素服从标准正态分布。 

==mat=X.T.dot(X)== X.T是X的转置，并与X相乘结果保存在mat（2维数据也即矩阵）。

矩阵的逆 ==inv== 、行列式值 ==det==、特征值和对应的特征向量==eig== 以及对矩阵进行奇异值分解 ==svd== 


## 其他
#### 数组创建
调用array的时候传入多个数字参数，而不是提供单个数字的列表类型作为参数。
```python
# 错误
a = np.array(1,2,3,4)
# 正确
a = np.array([1,2,3,4])
```

序列转换为二维数组
```python
b = np.array([1.5,2,3,(4,5,6)])
#b为
#array([[1.5,2,3],[4,5,6]])
#
```
**基本操作**
数组上的算数运算符会应用到元素级别
乘积运算符 * 在numpy数组中按元素进行运算
矩阵乘积可以使用 @ 运算符 或 dot函数或方法执行
![](../Python%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/images/022-09-24%20210742.gif)

```python
b = np.arange(12).reshape(3,4)
#b为
#array([[0,1,2,3],[4,5,6,7],[8,9,10,11]])

b.sum(axis=0)
#array([12,15,18,21]) 每一列的和

b.min(axis=1)
#array([0,4,8]) 每一行的最小值
```

#### 通函数
sin,cos,exp,在numpy中，这些被称为通函数，这些函数在数组上按元素进行运算，产生一个数组进行输出

**索引、切片和迭代**
- 多维的数组每个轴可以有一个索引，这些索引以逗号分隔的元组给出
# 当提供的索引少于轴的数量时，缺失的索引被认为是完整的切片
- 迭代  对多维数组进行 迭代 是相对于第一个轴完成的
（如果想要对数组中的每个元素执行操作，可以使用flat属性，该属性是数组的所有元素的迭代器）

**形状操纵，改变数组的形状**
一个数组的形状是由每个轴的元素的数量决定的

以下个命令都返回一个修改后的数组，但不会更改原数组
# ravel()
# reshape()

---
# ndarray.resize方法
会修改数组本身