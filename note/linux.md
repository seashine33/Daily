# 命令行命令
文件大小  
sudo du -sh *

文件夹下文件数目（不包括目录）
ls -l |grep "^-" | wc -l

统计当前目录下文件的个数（包括子目录）
ls -lR| grep "^-" | wc -l

复制文件夹下前一千个文件  
ls |head -n 1000 |xargs -i cp {} /temp


# zip  
## 解压
> tar -xvf FileName.tar  
> tar -zxvf FileName.tar.gz  
> unzip FileName.zip  
## 压缩
> tar -cvf FileName.tar DirName  
> gzip FileName // 只能压缩文件
> zip -r FileName.zip DirName

# 下载  
wget -P ~/code/Human-Trajectory-Prediction-via-Neural-Social-Physics-main/data/SDD_ini/annotations -c http://vatic2.stanford.edu/stanford_campus_dataset.

# 换源
## pip
-i https://pypi.tuna.tsinghua.edu.cn/simple  
-i https://mirrors.aliyun.com/pypi/simple/  
-i https://pypi.doubanio.com/simple  
## Conda
https://mirror.tuna.tsinghua.edu.cn/help/anaconda/
## ubuntu
https://blog.csdn.net/xiangxianghehe/article/details/122856771

# 添加环境变量  
export PATH=$PATH:/opt/au1200_rm/build_tools/bin

# screen
```
# 创建screen
screen -S name
# 列出所有screen
screen -ls
# 退出当前screen
ctrl+a+d
# 结束screen
exit
# 重新连接screen
screen -r ID
# 将Attatched设置为Detached，只有detached才能连上
screen -d name
# 结束进程
kill -9 
```

# 文件阅读器  
gedit

# wsl
https://learn.microsoft.com/zh-cn/windows/wsl/
```
wsl -l    //查看所有环境
wsl -l -v    //查看当前已有环境对应的发行版本
wsl --set-default-version <Version#>    //设置默认版本 1或2
wsl --list --online                    //查看所有可下载系统版本
wsl --install -d <DistributionName>    //安装指定版本的子系统
wsl --unregister <DistributionName>    //卸载指定子系统


//wsl2改变系统位置
//查看WSL分发版本
wsl -l --all -v
//导出分发版为tar文件到E盘 
wsl --export Ubuntu-18.04 D:\wsl-ubuntu18.04.tar
//wsl --export Ubuntu D:\wsl-ubuntu.tar
//注销当前分发版
wsl --unregister Ubuntu-18.04
//wsl --unregister Ubuntu
//重新导入并安装WSL在E:\wsl-ubuntu20.04
wsl --import Ubuntu-18.04 D:\wsl-ubuntu18.04 D:\wsl-ubuntu18.04.tar --version 2
//wsl --import Ubuntu E:\wsl-ubuntu E:\wsl-ubuntu.tar --version 2
//设置默认登陆用户为安装时用户名
ubuntu1804 config --default-user qh
删除tar文件(可选)
del D:\wsl-ubuntu18.04.tar
del E:\wsl-ubuntu.tar

 wsl --set-default <Distribution Name> //设置默认 Linux 发行版
 
 //释放空间
 wsl --shutdown
 # open window Diskpart
> diskpart
# 选择虚拟机文件执行瘦身
> select vdisk file="D:\WSL\docker-desktop-data\ext4.vhdx"
> attach vdisk readonly
> compact vdisk
> detach vdisk
> exit
```