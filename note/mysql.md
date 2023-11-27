```
sudo vi /etc/mysql/mysql.conf.d/mysqld.cnf
skip-grant-tables
/etc/init.d/mysql restart
```


```
// 以root用户登录
mysql -u root -p
```



```
// 修改密码
use mysql;
select user,host from user;
ALTER USER 'root'@'%' IDENTIFIED WITH mysql_native_password BY '@Whut123';
select user,plugin from user where user='root';
flush privileges;
```


```
use mysql;	// 切换库


show databases;		//查看所有库
show tables;		//查看库中所有表
select user from user;	//查看所有用户
desc 表名; 		//查看表的结构
select * from 表名; 	//查看表中数据


create user 'qh'@'%' identified by '@Whut123';	//新建用户
grant all privileges on *.* to 'testuser'@'%';	//添加全部权限
create database 库名; 				//建库
create table if not exists user(
  id int auto_increment, 
  ps int not null,
  primary key(id)
);						//建表
insert into 表名 values (,);			//往表中加入记录


drop database 库名;  				//删库
drop table 表名；  				// 删表
delete from 表名;  				//清空表
DELETE FROM user Where User='ding' and Host='%';// 删除用户
```