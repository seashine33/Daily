# 进度

## 25/5/29
* 下载了RT-Thread Studio，可以基于芯片生成代码，该软件也可以调用STM32CubeMX对外设进行配置
* 找到了正点原子板子的代码
  * https://github.com/RT-Thread/rt-thread/tree/master/bsp/stm32/stm32f103-atk-warshipv3
  * 但入门也可以不需要使用实际的板子
## 25/6/2
* [初识RT-Thread](https://www.rt-thread.org/document/site/#/rt-thread-version/rt-thread-standard/tutorial/quick-start/stm32f103-simulator/stm32f103-simulator)
* 下载代码: rtthread_simulator_v0.1.0
  * 在main.c中展示了一个跑马灯例子
  * 线程
  * 动态内存
  * 事件
  * 空闲任务钩子
  * 关闭中断以访问全局变量
  * 邮箱
  * 内存池
  * 消息队列
  * 互斥锁
  * 互斥量：高优先级需要一互斥量在低优先级手上，将该低优先级线程提升到与高优先级线程相同
    * 低优先级释放后，会回到原先优先级吗
  * 
  * 
* 下载了 RT-Thread env  