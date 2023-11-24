# 基本命令
```
git status  
git init  
git add                     // 添加文件到临时缓冲区  
git commit -m "text commit"  
git log  
git branch                  // 查看分支  
git branch a                // 创建a分支  
git branch -b a             // 创建分支的同时切换到分支  
git branch -d a             // 删除a分支，有事删不了，如没有合并到主分支  
git branch -D a             // 强制删除a分支  
git checkout master         // 切换到master分支  
git merge a                 // 将a分支合并到mester分支  
git tag                     // 查看标签  
git tag v1.0                // 为当前分支打标签  
```


# 上传
```
git commit -m "说明"  
git push origin main  
```

# 下载
```
git clone uml.git  
```

# 代理(修改，取消，查看)
```
git config --global http.proxy http://127.0.0.1:7890  
git config --global https.proxy http://127.0.0.1:7890  
git config --global --unset http.proxy  
git config --global --unset https.proxy  
git config --global --get http.proxy  
git config --global --get https.proxy  
```

# confog信息
```
[user]  
    name = XXX(自己的名称)  
    email = XXXX(邮箱)  
```