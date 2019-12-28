# 因子分解机(Factorization Machine, FM)

* 核心思想: 二阶特征组合

## 特征组合模型的进化角度

* 线性模型 LR
    * $ \hat{y} = w_0 + \sum_{i=1}^n{w_i x_i} $
    * 优点:简单、效率高
    * 缺点:难以捕获组合特征（需要人工特征组合，费时费力）

* 线性模型的改进: 直接在计算公式里加入二阶特征组合（类似多项式核SVM）
    * $ \hat{y} = w_0 + \sum_{i=1}^n{w_i x_i} + \sum_{i=1}^n\sum_{j=i+1}^n{w_{i,j}x_i x_j} $
    * $x_i x_j$理解为两种特征组合出来的特征，同时拥有这两种特征的样本可能很少，所以存在$w_{i,j}=0$的情况
    * 优点:直接将特征两两组合引入模型
    * 缺点:组合特征的泛化能力弱（当组合特征$x_i x_j = 0$时,$w_{i,j} = 0$）,不适用于大规模稀疏特征

* FM: 对每个特征学习一个`k维的稠密向量`,组合特征$x_i x_j$的权值用特征对应特征向量的内积表示
    * $ \hat{y} = w_0 + \sum_{i=1}^n{w_i x_i} + \sum_{i=1}^n\sum_{j=i+1}^n{<v_i, v_j>x_i x_j} $
    * 本质:做特征的embedding
    * 和DNN排序模型的区别:只是少了MLP，直接对多阶特征非线性组合建模
    * FM模型`泛化能力强`的根本原因:学习了特征的Embedding，即计算了特征之间的相关性作为两两特征组合的权值，即使没有包含这样组合特征的样本，也能通过内积算出这个新特征组合的权重

## 协同过滤模型（MF, Matrix Factorization, 矩阵分解）的进化角度

## 参考
[1] 原论文[Rendle, Factorization Machines, 2010](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
[2] 知乎: [推荐系统召回四模型之：全能的FM模型](https://zhuanlan.zhihu.com/p/58160982)