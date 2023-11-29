# [自动驾驶决策风险性评估文章(Risk Assessment)](https://zhuanlan.zhihu.com/p/653522020)
风险评估类型：异常车辆轨迹风险评估、碰撞等交通事故风险评估、激进驾驶风险评估、意外交通拥堵风险评估。  

不确定的因素：
> 数据的不确定性（数据分布问题->长尾现象）、知识的不确定性（数据是不完整的、有噪声的、不和谐的或多模态的）  
> 
> 基于深度学习的方式，对周围车辆和自身车辆路径进行编码，在用图卷积、attention等计算自身车辆与周围车辆的交互，最后联合两种信息进行解码，预测未来的路径或驾驶动作。  
> 基于强化学习的方法，学习自身车辆信息以及周围车辆的关系，再通过一定的学习策略（分布式强化学习、集成强化学习等）在仿真里设定相应的奖励函数，通过训练，学习到好的驾驶策略。  
> 
> 以上两种方法都可能涉及到CRF场、CVaR评估、驾驶缝补的均值和方差、RPF计算不确定性的驾驶策略。

推荐了几篇综述
> 2021年 [深度学习中的不确定性量化综述：技术、应用和挑战](./paper/A%20review%20of%20uncertainty%20quantification%20in%20deep%20learning.md)  
> 2022年 [自动驾驶风险评估方法综述](./paper/Risk_Assessment_Methodologies_for_Autonomous_Driving_A_Survey.md)  
> 2023年《Uncertainties in Onboard Algorithms for Autonomous Vehicles: Challenges, Mitigation, and Perspectives》

