# 碰撞风险评估仿真程序

* 仿真软件: sumo-1.15.0
  * 下载地址: https://sumo.dlr.de/releases/1.15.0/
* python环境
  * sumolib, traci等包

## 文件介绍
* scene_X文件夹
  * 为sumo软件需要的仿真文件，主要用于搭建场景
* basic.py
  * 主要用于绘画风险场
* sumo.py
  * 调用sumo进行仿真后绘图
  * 其中 file_base 为路径，得修改
  * 场景A为跟车场景
  * 场景B为切入场景