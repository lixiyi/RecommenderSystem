# Transformer里的位置编码

## 为什么需要
* RNN是顺序输入的，所以RNN会有位置感
* TF则完全没有位置信息，因而需要添加位置信息
* 第一个想法：
    * 用[0, 1]范围内的数字作为位置标记，0当作开头，1作为结尾
    * 问题：无法知道一个句子里有多少词，因而不同的句子间这样的位置信息不是一致的
* 第二个想法：
    * 用1, 2, 3等数字作为位置标记，当序列特别长的时候数字会特别大，此外对于那些少见的很长的序列泛化性差
* `所以pos emb必须满足`
    * 每个时间步值独一无二
    * 任意两个相同长度词之间的值相同
    * 为了具备更好的泛化性，数值范围得是有限的、确定的
## Pos Embedding
* 特点
    * 是向量，而非单个数字
    * 强化输入，而非修改模型
* 公式
    * t代表第t个位置
    * p_t是第t个位置的pos emb，维度为d，i是其中的一个维度
    * pos emb的偶数位 i = 2k
        * $ p_t^{(i)} = sin(w_k \cdot t) $
    * pos emb的奇数位 i = 2k + 1
        * $ p_t^{(t)} = cos(w_k \cdot t) $
    * $ w_k = \frac{1}{10000^{\frac{2k}{d}}} $ 
        * i越大，k越大，$ w_k $ 越小
        * 所以沿着d的方向 $ w_k $ 会越来越小, 也就是函数的周期变大，频减慢
* 由来
    * 数字的二进制最低位，0 / 1的周期变化频率就是逐渐减慢的
    * 如果直接用二进制位去做，float太耗空间了，所以采用二进制的连续版本，cos和sin
* 使用
    * pos emb dim = word emb dim
    * pos emb + word emb

## 相对位置
* 用上正弦曲线位置编码后，模型会自动关注相对位置
* 证明：rotate matrix, 和t无关
* 正弦曲线位置编码的邻居对称

## FAQ
* 为啥是add不是concat？
    * 没必要纠结，add更省空间
* 到了model的顶层，位置信息会消失嘛？
    * 不会，TF用了残差连接
* 为啥要同时用sin和cos
    * 只有同时用了才能保证sin(x+k)和cos(x+k)都是sin(x)和cos(x)的线性函数，只用单个做不到
    * 每一对sin和cos可以看作不同挪动频率的时钟

## Reference
[https://kazemnejad.com/blog/transformer_architecture_positional_encoding/](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)