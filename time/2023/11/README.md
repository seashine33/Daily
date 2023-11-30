# 2023.11.21 周二
上午九点起来的, 做了个leetcode, 然后把USMART看了会, 但没实际上手改, 把想看的硕士论文摆出来, 准备下午看。  
下午三点多来的实验室, 主要学了一下github的使用, 嗯还是没开始写毕业论文。  
晚上吃了个韩式炸鸡，然后看了个维多利亚3的视频，我就不想学习了。  

# 2023.11.22 周三
钢铁雄心4成功顶住了日本，没有丢过北京，就是一开始没有灭掉陕西，到后来打的时候苏联直接跟我宣战，没办法从苏联那进口钢材，海运又被封锁，所以后面的装备跟不上了。

# 2023.11.23 周四
做了每日一题，是一道字符串匹配，比较简单。  
然后就是把Git命令熟悉了一下，然后代理什么的也弄了一下，应该是能正常用了，添加了一个关于git的node。  
晚上去梦时代那边和源一起逛了逛，然后就回寝室了。  
把P社的启动器更新好了，后面可以正常添加mod了。  
晚上对面课题组吵架，看起来是关系缓和了。  

# 2023.11.24 周五
首先分析为什么QCNet的上传结果不好，我发现有一定的过拟合了，然后我试着把靠近中间的结果上传试试，是不是比训练后期的模型参数要好。  
不行，epoch42的结果是7.22，epoch36结果是8.38，不是过拟合。我再试个epoch 56，结果是7.08，这就很尴尬，像是没训练完一样。  
今天的机会用完了，还剩一个epoch52没试，明天再试试。  

今天的每日一题是两数之和，很简单。

下午把年终总结需要的文档写一下。在写年终总结的顺便想想我写的解码部分的逻辑，首先精预测部分肯定是多模态，肯定有根据模态数对query进行扩充，那粗预测部分呢？  
精预测要使用航点特征来作为query，那航点特征，等等哈，如果粗预测部分就是多模态航点，那我精预测部分就不需要多模态扩充了，也不需要删点，直接就使用多模态数量的航点特征。嗯，逻辑没问题。

写了一个push.bat，是win下的脚本，用来记录上传的命令。

元气骑士前传上线，不过服务器一直在炸。

# 2023.11.25 周六
醒来已经十一点。  
钢铁雄心又试了几把校长，好难啊，荡平军阀桂系最好是不服，但发现48个四步师也没法在七七之前拿下桂系，有点骑虎难下的意思。再去研究一下教程。  
嗯，看别人打真简单啊。  


# 2023.11.27 周一
epoch52, 结果为7.36，难绷。  
|batch|FDE|
| --- | --- |
|36|8.38|
|42|7.22|
|52|7.36|
|56|7.08|

然后我检查是不是val代码的问题，然后果然，算的FDE等都是所有车辆的数据，显然只用预测target。  
但也不能盲目乐观，因为结果没有很差，第一个epoch的FDE就是1.5了，还是只用半张卡跑的。  
所以就算是有这方面的问题，也不是关键，要知道test的结果可是都在6/7之间徘徊的。比HiVT的1.228不知道差哪去了。  
晚上吃了个冰龙。  
再次开始着手毕业的事。  
今天的每日一题是单调栈。  

# 2023.11.28 周二
卡空出来了，再跑一次batch为32的实验。  
昨天batch为16的跑了14个epoch，FDE到了1.08，算是很好了，HiVT最低也只用0.95。拿这个做个test，结果是9.72，一样的不正常。  
- [ ] 把相关超参数全部调成初始值，再做实验，看是不是过拟合，不管是什么原因的过拟合。

--train_batch_size 4  
--val_batch_size 4  
--test_batch_size 4  
--devices 8  
--dataset argoverse_v2  
--num_historical_steps 50  
--num_future_steps 60  
--num_recurrent_steps 3  
--pl2pl_radius 150  
--time_span 10  
--pl2a_radius 50  
--a2a_radius 50  
--num_t2m_steps 30  
--pl2m_radius 150  
--a2m_radius 150  

开始看毕业第二个点。  

下午去打了个台球。

# 2023.11.29 周三
- [ ] 除了调参以外，还可以把计算loss的过程改为把预测值旋转平移到世界坐标系，与真值求欧氏距离。  
- [x] Risk Assessment 帖子推荐的第一篇论文了解了一下，这是一篇介绍不确定性相关研究的综述，总的来说就是看不懂。
- [ ] Risk Assessment 帖子推荐的第一篇论文是讲AV的风险评估(RA)的，重点看看。

# 2023.11.30 周四
- [x] 把val修改为只计算target车，训练结果正常了很多，FDE在val的最低为0.948左右。已经训练出60个epoch，其中epoch 54的值最低，拿它来test一下，结果是6.85。
- [x] 发现了一个华点，为什么监督真值与预测值距离的loss是个负的？可能这就是反复观看，每次看都有新感觉吧。  

这是 LaplaceNLLLoss
```Python
loc, scale = pred.chunk(2, dim=-1)  # [A, 60, 1], 位置, [A, 60, 1], 比例
scale = scale.clone()
with torch.no_grad():
    scale.clamp_(min=self.eps)      # 规定下限
nll = torch.log(2 * scale) + torch.abs(target - loc) / scale
```

这是解码得到scale的过程，elu激活函数我还能理解，为什么还要累加？
```Python
scale_propose_pos = torch.cumsum(
    F.elu_(
        torch.cat(scales_propose_pos, dim=-1).view(-1, self.num_modes, 
            self.num_future_steps, self.output_dim), # Tensor(A, 6, 30, 2)
        alpha=1.0) + 1.0,   # elu激活函数, x<=0时, exp(x), x>0时, x+1
    dim=-2) + 0.1       # 数值沿时间维度累加，Tensor(A, 6, 30, 2)
```

reg_loss() 就是 LaplaceNLLLoss，这是最外层算loss的部分，这里没有用最常见的两点距离计算loss，而是xy两轴分别算loss，然后再求和。
```Python
reg_loss_propose = self.reg_loss(
        traj_propose_best,
        t[..., :self.output_dim + self.output_head]
    ).sum(dim=-1) * reg_mask  # sum(Tensor(A, 30, 2), -1)->Tensor(A, 30)
reg_loss_propose = reg_loss_propose.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)              # Tensor(30)
reg_loss_propose = reg_loss_propose.mean()  # Tensor(1), 提议轨迹loss
```

把超参数全部调整成原先的数值了，准备完整地再练一遍，这不得跑十几天。

接着看论文吧。