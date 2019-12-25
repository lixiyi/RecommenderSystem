# 召回

目标：初步筛选用户感兴趣的item，通常多路并发召回，各路召回互不影响

## 传统方法: 协同过滤
* iiCF
    * 用户历史item中找最相似的item,将该用户自身评分填入
    * 缺点：
        * 候选item限定在用户的历史行为类目中
        * 难以结合候选item的Side Information
        * 发现性弱
        * 对长尾商品的效果差
        * 容易导致推荐系统出现`越推越窄`的问题
* uuCF
    * 相似用户对item的评分(各自归一化)均值填入
* 基于模型的协同过滤
    * MF矩阵分解

## 单Embedding
* 将user和item通过DNN映射到`同一个低维度向量空间`中，然后通过高效的`检索方法`去做召回
* 双塔模型
* YoutubeDNN

## 多Embedding
* Multi-Interest Network with Dynamic Routing (MIND)
    * 只用一个embedding向量来表示用户的兴趣其表征能力是远远不够的，MIND模型通过引入capsule network的思想来解决输出多个向量embedding的问题 (`Atteintion`)

## Graph Embedding
* 阿里Graph Embedding with Side information
* GraphSAGE：Inductive representation learning on large graphs

## 结合长短期兴趣
* SDM: Sequential Deep Matching Model for Online Large-scale Recommender System
* Next Item Recommendation with `Self-Attention`

## 深度树匹配
* 阿里TDM：Tree-based Deep Model


# 参考

[1] [知乎：谈谈推荐场景中召回模型的演化过程](https://zhuanlan.zhihu.com/p/97821040)