---
title: 线性代数
---

# 线性代数

因为数学库有好多现成的，这里只是写着玩，方便理解。

**1. dot product:** 投影长度

$\vec{\mathbf{a}}$ * $\vec{\mathbf{b}}$ = $|a| * |b| * \cos\theta = \sum\limits_{i=1}^n a_i*b_i = \vec{\mathbf{a}}^T
\vec{\mathbf{b}}$

**2. cross product:** 有向面积, 右手定则

$$
\vec{\mathbf{a}} \times \vec{\mathbf{b}} = |a|*|b|*sin\theta*\hat{\mathbf{n}}
$$

若

$ \vec{\mathbf{a}} \times \vec{\mathbf{b}} = \vec{\mathbf{c}}, \vec{\mathbf{a}} = (a_x, a_y, a_z)^T, \vec{\mathbf{b}} = ( b_x, b_y, b_z)^T $

, 则:

$\vec{\mathbf{c}} = \begin{pmatrix} a_y*b_z - a_z*b_y \\\\ a_z*b_x - a_x*b_z \\\\ a_x*b_y - a_y*b_z \end{pmatrix}$

**3. 旋转矩阵**

`基`$[\vec{e_1}, \vec{e_2}, \vec{e_3}]$定义一个坐标系，该`基`旋转变为$[\vec{e_1}', \vec{e_2}', \vec{e_3}']$。
则向量$\vec{a} = \begin{pmatrix}a_1\\ a_2\\\\ a_3 \end{pmatrix}$, 在新坐标系内的表示变为$\vec{a}' = \begin{pmatrix}a_1' \\\\ a_2' \\\\
a_3' \end{pmatrix}$ （注意$\vec{a}'$并没有经这坐标变换),
则：

$[\vec{e_1}, \vec{e_2}, \vec{e_3}] \begin{pmatrix} a_1 \\\\ a_2 \\\\ a_3 \end{pmatrix} = [\vec{e_1}', \vec{e_2}', \vec{e_3}'] \begin{pmatrix} a_1' \\\\ a_2' \\\\ a_3' \end{pmatrix} $

则：

$ \begin{pmatrix} a_1 \\ a_2 \\ a_3 \end{pmatrix} = \begin{pmatrix} e_1^T e_1' & e_1^Te_2' & e_1^T e_3' \\\\ e_2^T e_1' & e_2^Te_2' & e_2^T e_3' \\\\ e_3^T e_1' & e_3^Te_2' & e_3^T e_3' \end{pmatrix} \begin{pmatrix} a_1' \\\\ a_2' \\\\ a_3' \end{pmatrix} = \vec{R}\vec{a}' $

SO(3)特殊正交群，(Special Othogonal Group), 正交矩阵（行列式为1)：

SO(n) =

$
\{\vec{R} \in R^{n \times n} | \vec{R}\vec{R}^T= \vec{I}, det(\vec{R}) = 1 \}
$

**4. 欧氏变换**

欧氏变换 = 旋转 + 平移

$\vec{a}' = \vec{R} \vec{a} + \vec{t}$

**5. 齐次坐标**

\\[
\begin{pmatrix} \vec{a}' \\\\ 1 \end{pmatrix} =
\begin{pmatrix} \vec{R} & \vec{t} \\\\
\vec{0}^T & 1\end{pmatrix}
\begin{pmatrix} \vec{a} \\\\ 1\end{pmatrix} =
\vec{T}
\begin{pmatrix} \vec{a} \\\\ 1\end{pmatrix}
\\]

T属于特殊欧氏群(Special Euclidean Group), SE(n)

SE(3) =

$ \{ \vec{T} = \begin{pmatrix} \vec{R} & \vec{t} \\\\ \vec{0}^T & 1\end{pmatrix} \in R^{4 \times 4} | \vec{R} \in SO(3), t \in R^3 \} $

**6. 表**

---

| 变换名称           | 归类               | 矩阵形式                                                             | 自由度 | 不变性质             | 备注                                                           |
| -------------------- | -------------------- | ---------------------------------------------------------------------- | -------- | ---------------------- | ---------------------------------------------------------------- |
| 欧氏变换           | 线性变换，刚性变换 | $\begin{pmatrix} \vec{R} & \vec{t} \\\\ \vec{0}^T &1 \end{pmatrix}$  | 6      | 长度、夹角、体积     | 旋转＋平移 自由度：旋转3, 平移3                            |
| 相似变换           | 线性变换           | $\begin{pmatrix} \vec{sR} & \vec{t} \\\\ \vec{0}^T &1 \end{pmatrix}$ | 7      | 体积比               | 旋转＋平移＋缩放自由度：旋转3, 平移3, 均匀缩放1            |
| 仿射变换(正交投影) | 线性变换           | $\begin{pmatrix} \vec{A} & \vec{t} \\\\ \vec{0}^T &1 \end{pmatrix}$  | 12     | 平行性，体积比       | $\vec{A}$只要求可逆，不要求正交自由度：旋转3, 平移3, 缩放1 |
| 射影变换           | 线性变换           | $\begin{pmatrix} \vec{A} & \vec{t} \\\\ \vec{a}^T & v \end{pmatrix}$ | 15     | 接触平面的相交和相切 | 自由度：旋转3, 平移3, 缩放1                                    |

**7. 图形学中常见变换

| 符号(Notation)                  | 名字(Name)                     | 特性                                           |
|-------------------------------|------------------------------|----------------------------------------------|
| $\vec{T}(t)$                  | 平移矩阵                         | Affine                                       |
| $\vec{R_x}(\rho)$             | 旋转矩阵                         | 线x轴旋转$\rho$弧度。Orthogonal & Affine            |
| $\vec{R}$                     | 旋转矩阵                         | Orthogonal & Affine                          |
| $\vec{S}(s)$                  | 缩放矩阵                         | x, y, z同时均匀缩放s。Affine                        |
| $\vec{H}_{ij}(s)$             | 错切矩阵(shear matrix)           | 使用系统s来相对于分量j错切（推移）分量i,$i,j \in \{ x, y, x\}$ |
| $\vec{E}(h,p,r)$              | 欧拉变换(Euler Transform)        | yaw, pitch, roll  Orthogonal & affine        |
| $\vec{P}_o(s)$                | 正交投影(orthogonal projection)  | Affine                                       |
| $\vec{P}_p(s)$                | 透视投影(perspection projection) | ..                                           |
| $slerp(\hat{q}, \hat{r}, t) $ | 线性插值变换(slerp transform)      | 对四元数$\hat(q), \hat(r)$用参数t插值得到的新四元数          |

| 符号(Notation)                  | 名字(Name)                     | 特性                                           | 表示                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
|-------------------------------|------------------------------|----------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| $\vec{T}(t)$                  | 平移矩阵                         | Affine                                       | $\begin{pmatrix} I^3 & \vec{t} \\\\ 0^t & 1 \end{pmatrix}$                                                                                                                                                                                                                                                                                                                                                                                                 |
| $\vec{R_x}(\rho)$             | 旋转矩阵                         | 线x轴旋转$\rho$弧度。Orthogonal & Affine            | $\vec{R}_x(\phi) = \begin{pmatrix} 1 & 0 & 0 & 0 \\\\ 0 & cos\phi & -sin\phi & 0 \\\\ 0 & sin\phi & cos\phi & 0 \\\\ 0 & 0 & 0 & 1 \end{pmatrix}$ $\vec{R}_y(\phi) = \begin{pmatrix} cos\phi & 0 & sin\phi & 0 \\\\ 0 & 1 & 0 & 0 \\\\ -sin\phi & 0 & cos\phi & 0 \\\\ 0 & 0 & 0 & 1 \end{pmatrix}$ $\vec{R}_z(\phi) = \begin{pmatrix} cos\phi & -sin\phi & 0 & 0 \\\\ sin\phi & cos\phi & 0 & 0 \\\\ 0 & 0 & 1 & 0 \\\\ 0 & 0 & 0 & 1 \end{pmatrix}$ |
| $\vec{R}$                     | 旋转矩阵                         | Orthogonal & Affine                          |                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| $\vec{S}(s)$                  | 缩放矩阵                         | x, y, z同时均匀缩放s。Affine                        | $\vec{S}(\vec{s}) = \begin{pmatrix} s_x & 0 & 0 & 0 \\\\ 0 & s_y & 0 & 0 \\\\ 0 & 0 & s_z & 0 \\\\ 0 & 0 & 0 & 1\end{pmatrix}$                                                                                                                                                                                                                                                                                                                             |
| $\vec{H}_{ij}(s)$             | 错切矩阵(shear matrix)           | 使用系统s来相对于分量j错切（推移）分量i,$i,j \in \{ x, y, x\}$ | $\vec{H}_{xz}(s) = \begin{pmatrix} 1 & 0 & s & 0 \\\\ 0 & 1 & 0 & 0 \\\\ 0 & 0 & 1 & 0 \\\\ 0 & 0 & 0 & 1 \end{pmatrix}$                                                                                                                                                                                                                                                                                                                                  |
| $\vec{E}(h,p,r)$              | 欧拉变换(Euler Transform)        | yaw, pitch, roll  Orthogonal & affine        |                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| $\vec{P}_o(s)$                | 正交投影(orthogonal projection)  | Affine                                       |                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| $\vec{P}_p(s)$                | 透视投影(perspection projection) | ..                                           |                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| $slerp(\hat{q}, \hat{r}, t) $ | 线性插值变换(slerp transform)      | 对四元数$\hat(q), \hat(r)$用参数t插值得到的新四元数          |                                                                                                                                                                                                                                                                                                                                                                                                                                                            |

