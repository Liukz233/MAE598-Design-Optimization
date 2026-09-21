# 作业二：MBB 梁分析中的病态问题

**课程：** MAE 598/494 Design Optimization

**团队：** OptiForge

**成员：** Kangzheng Liu

**作业要求：** [Project 2: Ill-Conditioned Optimization](https://designinformaticslab.github.io/DesignOptimization2025/project2.html)

## 1. 问题背景与研究动机

结构工程师使用有限元分析预测构件在载荷作用下的变形。在拓扑优化中，每次材料分布发生变化，都需要重复这一计算。更细的网格能够描述更多结构细节，但也可能使平衡方程更难通过迭代方法求解。

本项目采用[作业一](../../01_problem_formulation/report/report.md)中的半 MBB 梁。左上角施加竖直向下的单位载荷，左边界限制水平位移，右下角设置滚动支座。对于每个给定的材料布局，优化任务是求出使总势能最小的位移场。

![半 MBB 梁及其载荷与支承](../../01_problem_formulation/figures/problem_setup.png)

我们先研究网格加密对条件数和梯度下降收敛的影响，再比较共轭梯度法与 Jacobi 预条件的效果。作业一得到的优化材料布局作为第二个算例，用于考察实体和空域之间存在较大刚度差异时的求解表现。

## 2. 数学建模

梁的设计域为 120 × 40，厚度为 1，采用平面应力假设和四节点正方形单元。沿长度和高度方向的单元数满足 $n_x=3n_y$，单元边长为 $h=40/n_y$。网格加密过程中，几何尺寸、载荷和支承保持不变。

| 物理量 | 定义 | 数值或允许范围 |
|---|---|---|
| $\mathbf u\in\mathbb R^{n_d}$ | 连续的节点位移变量，单位为长度 | $n_d=2(n_x+1)(n_y+1)$；支座处位移约束见下文 |
| $\mathbf u_f\in\mathbb R^{n_f}$ | 施加支承条件后的自由位移变量 | $n_f=n_d-(n_y+2)$；无额外上下界 |
| $\rho_e$ | 给定的无量纲单元密度 | $0\leq\rho_e\leq1$ |
| $\mathbf F$ | 节点载荷向量 | 左上角节点处的竖直向下单位载荷 |
| $E_0,E_{\min}$ | 实体与空域的杨氏模量，单位为力除以面积 | $E_0=1$，基准值 $E_{\min}=10^{-9}$ |
| $p,\nu$ | SIMP 惩罚指数与泊松比 | $p=3$，$\nu=0.3$ |

计算沿用作业一的归一化单位。材料刚度采用 SIMP 插值：

$$
E_e=E_{\min}+\rho_e^p(E_0-E_{\min}).
$$

由此组装得到刚度矩阵 $K$，优化问题为

$$
\begin{aligned}
\min_{\mathbf u\in\mathbb R^{n_d}}\quad
&\Pi(\mathbf u)=\frac12\mathbf u^{\mathsf T}K\mathbf u-\mathbf F^{\mathsf T}\mathbf u,\\
\text{s.t.}\quad
&u_x=0\quad\text{on the left edge},\\
&u_y=0\quad\text{at the lower-right corner}.
\end{aligned}
$$

两个约束分别表示左边界的水平位移为零、右下角的竖直位移为零。目标中的两项分别表示应变能和外载荷势能，单位均为力乘以长度。消去已知位移后，有

$$
\nabla\Pi(\mathbf u_f)=K_{ff}\mathbf u_f-\mathbf F_f,
\qquad H=\nabla^2\Pi=K_{ff}.
$$

因此，势能最小化给出平衡方程 $K_{ff}\mathbf u_f=\mathbf F_f$。正的材料模量与支承条件使 $K_{ff}$ 成为对称正定矩阵。约化后的问题是连续、确定性、无约束的严格凸二次优化问题，具有唯一的最优解。120 × 40 网格共有 9,880 个自由位移变量。

## 3. 病态性的来源

主要机制是弹性微分算子的离散化，对应作业要求中的 B 类。与局部快速变化的变形相比，平滑的整体变形具有较低的能量。随着网格加密，刚度矩阵所表示的变形模态范围不断扩大。

这一点可以通过满足支承条件的位移场 $v_x=0$、$v_y=1-x/120$ 说明。网格加密时，它的应变能保持不变，而节点位移值的平方和按 $h^{-2}$ 增长。因此，由 Rayleigh 商可得 $\lambda_{\min}\leq C h^2$。局部刚度项则为 $\lambda_{\max}$ 提供了一个与网格无关的正下界，所以在这一网格序列中，条件数至少按 $h^{-2}$ 的量级增长。

为判断求解困难是否仅由各变量的尺度差异引起，我们比较

$$
\kappa(K_{ff})=\frac{\lambda_{\max}(K_{ff})}{\lambda_{\min}(K_{ff})}
\quad\text{and}\quad
\kappa(J),\qquad
J=D^{-1/2}K_{ff}D^{-1/2},\quad D=\operatorname{diag}(K_{ff}).
$$

取均匀密度 0.5 时，结果如下：

| 网格 | 自由变量数 | 原始条件数 | Jacobi 缩放后的条件数 |
|---|---:|---:|---:|
| 12 × 4 | 124 | 15,543 | 12,446 |
| 24 × 8 | 440 | 59,263 | 51,807 |
| 48 × 16 | 1,648 | 228,461 | 212,348 |
| 120 × 40 | 9,880 | 1,404,156 | 1,363,443 |

![Jacobi 缩放前后，条件数随网格分辨率的变化](../figures/mesh_conditioning.png)

分辨率加倍时，条件数大约增大到原来的四倍。对角缩放带来的改善有限，且没有改变增长趋势。调整各变量的尺度后，求解困难仍然存在。

对于这些均匀网格，每个位移自由度的对角刚度来自一个、两个或四个相同单元的贡献。因此 $\kappa(D)\leq4$，并且

$$
\kappa(J)\geq\frac{\kappa(K_{ff})}{\kappa(D)}
\geq\frac{\kappa(K_{ff})}{4}.
$$

这一界限说明，Jacobi 缩放无法消除随网格加密而增强的病态性。

## 4. 病态性对收敛的影响

两个较小网格的完整特征值谱展示了不同方向上的曲率差异。对于较大的系统，则计算其最小和最大特征值。

![12 × 4 与 24 × 8 网格的特征值谱](../figures/eigenvalue_spectra.png)

梯度下降法（GD）按下式更新自由位移：

$$
\mathbf u_{k+1}=\mathbf u_k-\alpha(K_{ff}\mathbf u_k-\mathbf F_f),
\qquad
\alpha=\frac{2}{\lambda_{\max}+\lambda_{\min}}.
$$

对于正定二次问题，这一步长使最坏情况下的误差收缩因子最小。最慢方向的收缩因子为 $q=(\kappa-1)/(\kappa+1)$。当 $\kappa$ 很大时，$q$ 接近 1，误差下降缓慢。

所有方法均从零位移开始，采用相同的相对平衡残差：

$$
r_k=\frac{\|K_{ff}\mathbf u_k-\mathbf F_f\|_2}{\|\mathbf F_f\|_2}
\leq10^{-6}.
$$

确认收敛前，重新计算原系统的残差。比较方法包括 GD、Jacobi 缩放后的 GD、共轭梯度法（CG）和 Jacobi 预条件共轭梯度法（Jacobi-PCG）。GD 类方法最多更新 120,000 次，CG 类方法最多更新 5,000 次。

| 网格 | GD | Jacobi-GD | CG | Jacobi-PCG |
|---|---:|---:|---:|---:|
| 12 × 4 | 93,648 | 75,404 | 76 | 68 |
| 24 × 8 | 达到上限 | 达到上限 | 150 | 138 |
| 48 × 16 | 达到上限 | 达到上限 | 293 | 275 |
| 120 × 40 | — | — | 710 | 686 |

表中数值为达到规定容差所需的更新次数。“达到上限”表示更新 120,000 次后仍未达到容差；短横线表示未运行该方法。达到上限时，24 × 8 网格上 GD 和 Jacobi-GD 的残差分别为 $1.60\times10^{-3}$ 和 $9.24\times10^{-4}$；48 × 16 网格上则分别为 $1.66\times10^{-2}$ 和 $1.56\times10^{-2}$。

![24 × 8 网格上的残差收敛曲线及前 500 次更新的局部放大图](../figures/convergence_uniform_24x8.png)

即使在 12 × 4 网格上，GD 也需要 93,648 次更新。Jacobi 缩放将其减少到 75,404 次，这与它对条件数的有限改善一致。CG 仅需 76 次更新便达到相同容差。随着网格加密，CG 的迭代次数也会增加，但仍远少于 GD。

## 5. 改善方法及效果

CG 通过构造关于 $K_{ff}$ 共轭的搜索方向，处理位移变量之间的耦合。这减少了沿已搜索方向反复修正误差的情况。对于正定二次问题，CG 的收敛界依赖于 $\sqrt{\kappa}$，而固定步长 GD 则依赖于 $\kappa$。

Jacobi 预条件根据对角刚度缩放位移。令 $\mathbf z=D^{1/2}\mathbf u_f$，变换后的 Hessian 为 $J$。Jacobi-GD 在这些新坐标中执行梯度下降，并根据 $J$ 的最大和最小特征值选择步长。Jacobi-PCG 则将这一缩放与共轭搜索方向结合。

### 优化后的材料布局

作业一的最终布局中，刚度较高的承载路径周围分布着低密度区域。我们在 120 × 40 网格上固定这一密度场，通过改变 $E_{\min}$ 考察材料刚度差异的影响。

![作业一得到的优化材料布局](../../01_problem_formulation/figures/final_topology.png)

| $E_{\min}$ | 原始条件数 | Jacobi 缩放后的条件数 | Jacobi-PCG 更新次数 |
|---|---:|---:|---:|
| $10^{-3}$ | $2.310\times10^6$ | $1.127\times10^6$ | 1,229 |
| $10^{-6}$ | $1.281\times10^9$ | $1.132\times10^6$ | 1,216 |
| $10^{-9}$ | $1.188\times10^{12}$ | $1.132\times10^6$ | 1,201 |

随着空域刚度减小，原始条件数迅速增大。Jacobi 缩放消除了大部分额外的特征值跨度，使条件数保持在 $1.13\times10^6$ 附近，PCG 的迭代次数也较为接近。网格实验说明了问题固有的病态性，这一算例则展示了预条件处理额外材料尺度差异的效果。

在原始设定 $E_{\min}=10^{-9}$ 下，CG 更新 5,000 次后的残差为 $7.23\times10^{-3}$。Jacobi-PCG 在 1,201 次更新后达到 $9.27\times10^{-7}$。图中还给出了相对目标函数差

$$
g_k=\frac{\Pi(\mathbf u_k)-\Pi(\mathbf u^\star)}
{\Pi(\mathbf 0)-\Pi(\mathbf u^\star)},
$$

其中，$\mathbf u^\star$ 由稀疏直接求解器得到。代码使用 $\tfrac12(\mathbf u_k-\mathbf u^\star)^{\mathsf T}K_{ff}(\mathbf u_k-\mathbf u^\star)$ 计算分子。

![优化布局的平衡残差与相对目标函数差](../figures/convergence_optimized_120x40_Emin_1e-09.png)

参考柔顺度为 210.0406，与作业一一致。Jacobi-PCG 计算得到的柔顺度相对误差小于 $10^{-10}$。两个实验表明，CG 能显著减少细化均匀梁网格的求解迭代次数，而 Jacobi 预条件对于优化布局中的大幅材料刚度差异尤为有效。

## 6. 假设与简化

模型采用各向同性线弹性、小变形、平面应力、单位厚度和单一静载工况假设。每次求解过程中，密度保持不变。单位载荷用于确定归一化尺度；将其减小以满足小变形条件，不会改变条件数和相对误差的比较结果。

点载荷和单节点滚动支座沿用作业一的设置。本文研究这一离散模型的条件数及平衡求解器的收敛情况，不涉及应力限制、屈曲、非线性变形或局部应力的网格收敛性。计算改进以达到相同残差容差所需的迭代次数衡量。

## 7. 复现方法

使用 Python 3.12，在仓库根目录运行：

```bash
python -m pip install -r projects/02_gradient_descent/requirements.txt
python projects/02_gradient_descent/src/conditioning_demo.py
```

[脚本](../src/conditioning_demo.py)调用作业一的有限元函数并读取已保存的密度场，生成图表和数值结果。环境版本和随机种子记录在 [metadata.json](../results/metadata.json) 中。条件数保存在 [conditioning.csv](../results/conditioning.csv) 中，迭代次数、最终残差和误差保存在 [solvers.csv](../results/solvers.csv) 中。[结果目录](../results/)还包含采样后的收敛历史。

一个可以手算的小算例用于核对条件数和迭代次数。对于泊松比 $\nu=0.3$ 的单个实体正方形单元，仅保留上方两个节点的竖直位移，其余位移全部固定，得到

$$
H_{\mathrm{check}}=\frac1{91}
\begin{bmatrix}45&5\\5&45\end{bmatrix},
\qquad
\lambda_1=\frac{40}{91},\quad\lambda_2=\frac{50}{91},\quad\kappa=1.25.
$$

取 $\mathbf b=(1,0)^{\mathsf T}$ 和零初始位移时，采用最优固定步长的 GD 的相对残差为 $9^{-k}$，七次更新后达到 $10^{-6}$；CG 则需要两次。代码复现了这些数值，并检查了能量梯度、稀疏特征值计算及作业一的柔顺度。检查结果保存在 [verification.json](../results/verification.json) 中。

## 参考文献

1. MAE 598/494，[Project 2: Ill-Conditioned Optimization](https://designinformaticslab.github.io/DesignOptimization2025/project2.html)。
2. Kangzheng Liu，[Project 1: Formulation and Solution of a SIMP Topology-Optimization Problem](../../01_problem_formulation/report/report.md)。
3. E. Andreassen, A. Clausen, M. Schevenels, B. S. Lazarov, and O. Sigmund, “Efficient topology optimization in MATLAB using 88 lines of code,” *Structural and Multidisciplinary Optimization*, 43, 1–16, 2011. [doi:10.1007/s00158-010-0594-7](https://doi.org/10.1007/s00158-010-0594-7).
