# FFM （Field-aware FM）

* 能够意识到特征域(Field)的存在的FM模型
* FM : 特征组合时共享同一个Embedding，每个特征一个Embedding，特征不分特征域
* `FFM : 与不同的特征域里的特征组合时用不同的特征Embedding，特征域里的每个特征都有F-1个Embedding(一共n个特征,这n个特征分属于F个特征域)`
* 参数上，FM需要 $ n \times k $ 个参数，FFM需要$ (F-1) \times n \times k $ 个参数，FFM无法做类似FM的改写，所以它的计算复杂度是 $O(k\cdot n^2)$
* FFM参数数量过大，容易造成过拟合（一般在几千万训练数据规模下，k取8到10,训练集越小，k越小）


## 用FM/FFM做召回和排序的区别

* 排序阶段: 用户特征和item特征可以同时作为模型的输入（用户和要排序的Item都已知）
* 召回阶段: 要计算的item数据量巨大，且只有用户特征（即用户特征和Item特征不能同时作为模型的输入）
* 把召回阶段采用FFM这种排序模型看成一种受约束的排序,因为User和Item的特征不能同时输入
* `Hint: 离线存储和在线提取的用户Embedding, 其实就是FM里第一个平方项需要计算的, n个特征的Embedding乘上用户特征取值的累加和得到的向量V`


## 用FFM做召回模型

* 线上收集到的用户ctr来作为训练数据
* 把特征域划分为三个子集合:
    * 用户相关特征集合(User):用户历史行为类特征，如用户过去点击物品的特征，可以当作描述用户兴趣的特征，放入用户相关特征集合内
    * 物品相关特征集合(Item)
    * 上下文相关的特征集合(Context)
* 把User里的特征分为两个特征阈U1, U2; 把Item里的特征分为I1, I2, I3，则每个特征域里的特征都需要有4个Embedding(一共5个特征域，F = 5)

### User和Item`间`的特征Embedding内积计算

* $U_1$的3个Embedding表示为：$U_{11}, U_{12}, U_{13} $ (一共3个Item特征域)
* $I_1$的2个Embedding表示为：$I_{11}, I_{12}$ (一共2个User特征域)
* 对应的内积关系：$U_{11} \rightarrow I_{11}, U_{12} \rightarrow I_{21}, U_{13} \rightarrow I_{31}$, `I转置一下就和U内积关系对应上了`
* 因而User和Item所有特征域对应的Embedding都可以时先Concat再Flatten后存储：$ [ U_{11}, U_{12}, U_{13},U_{21}, U_{22}, U_{23} ], [ I_{11}, I_{21}, I_{31}, I_{12}, I_{22}, I_{32} ] $; 这样的内积计算和原本FFM公式是等价的:
    * $ <V_i, V_j>x_i x_j = (\sum_{f=1}^k V_{if}V_{jf})x_i x_j = \sum_{f=1}^k (V_{if}x_i \cdot V_{jf}x_j) = <V_i x_i, V_j x_j> $
    * `Hint: User与Item存储的时候就直接存储Concat + Flatten后的V_i X_i得到的Embedding即可`
* `勿混淆：Uij 都是同一个User特征的Embedding；Iij也都是同一Item特征的Embedding，只是不同的特征域而已`
* FFM的Embedding长度明显变得很长，若User 50个特征与，Item 50个特征域, k=10,就会有25000长度的Embedding,若要减小Embedding有两种方法:
    * 减小k, eg:k = 2或4,召回阶段不用特别准，但5000的Embedding依旧很长
    * 减少特征域的个数,受损严重，不能从根本上解决问题
* `注意这种Concat + Flatten的Embedding存储内容和FM的存储内容不同，FM存的是累加后的V向量，这里明显降低了计算量，而FFM存的是Concat后的向量，没有减少计算量`

### 加入User和Item`内部`的组合特征

* 内部组合特征计算方法可以仿照上一步，或严格按FFM的公式计算，将得到的score拼接到User和Item的Embedding末尾：1，User Score ; Item Score, 1, 即可同步完成内积运算
* 若只用FM/FFM模型做召回，用户侧内部的特征组合对于返回结果排序没有影响，所以可以不用加入；而物品侧内部特征之间的特征组合可能会对返回的物品排序结果有影响，可以考虑引入

### 加入一阶项

* 加入的方法和User与Item内部组合特征的方法相同，也可以把一阶项的特征拆开Concat到二阶项Embedding的末尾
* `一阶项对于最终效果有明显影响，所以在用FM/FFM做召回的时候，是需要将一阶项考虑进去的，这可能是个别一阶特征比较重要导致的`
* 如果是采用DeepFM模型，则FM部分是否保留一阶项对最终结果没有什么影响，因为DNN的隐层有效地将一阶项的作用吸收掉了

### 加入上下文特征Context

* 不太可能离线算好存起来直接使用的，而是可能用户每一次刷新都需要重新捕获的特征值，`动态性强`, 上下文特征有时是非常强的特征
* Context特征的向量分为三组：User、Item、Context内部
    * Context特征与User特征的内积<U, C>计算时，与上述User和Item内积计算的方法相同，`设Context侧只有一个特征`，则U1和U2需要各自增加一个和Context特征计算内积的Embedding：$ U_{14}, U_{24} $
    * Context特征与Item特征的内积<U, I> + <C, I>计算时，利用User与Item内积的计算过程，在User的末尾Concat上Context侧特征的Embedding, I1、I2和I3需要各自增加一个和Context特征计算内积的Embedding：$ I_{13}, I_{23}, I_{33} $
    * 上述两步计算的Score相加即可得到最终召回排序，`<U, C>在召回阶段对物品排序没有影响，在召回阶段可以不考虑`

### 提速策略

* 上述方法考虑了U,I,C之间特征的两两组合，一阶项一级U,I内部特征的两两组合，但生成的向量过长，有损Faiss检索的速度
* 并行拉取提速：将Embedding分割存储，各自计算内积后加和
    * 存在漏召回的情况，比如，综合总得分较高但每个片段的得分都不太高也不太低的物品没有被任何一个Faiss子数据库拉回来
* （FM + FFM）混合提速：仿照FM累加得到V向量的方式减小Embedding的大小
    * 若是完整实现FFM是不可能累加起来的，因为特征Embedding是一一对应的，没有公因子项可以提出
    * 若不遵守FFM规则（`介于FM和FFM之间的方法`）:
        * 在User侧针对同一Item特征Embedding累加：$ [U_{11} + U_{21}, U_{12} + U_{22}, U_{13} + U_{23}] $
        * 在Item侧同一特征的Embedding累加：$ [I_{11} + I_{12}, I_{21} + I_{22}, I_{31} + I_{32}] $

## 用FFM统一召回
* `FFM是很难引入协同特征，除非事先通过其它方法对ID进行协同embedding编码`
    * FM可以，协同特征如MF,MF可以看作特征只有ID的FM，而FM是FFM的特例，但FFM需要对每个特征都有（F-1）个Embedding，这对于ID来说不现实


## 参考

* [1] 知乎：[推荐系统召回四模型之二：沉重的FFM模型](https://zhuanlan.zhihu.com/p/59528983)