# 标题
当让就是几个#就是几级标题

# 段落
空一行

再一行

# 换行
后面加两空格  
再加一行

# 强调
可以在语句中加入**加粗部分**。  
也可以加入*斜线*

# 引用
在段落前加>符号
> 这是第一句引用
> 这是第二句引用
>
> 这是第三句引用
>> 这是嵌套引用

# 列表
1. 第一部分
   1. 1.1部分
      1. 1.1.1部分
   2. 1.2部分
2. 第二部分
3. 第三部分

# 代码
如果是段落中的代码，可以用`int a = 10;`表示出来。
如果是整段的代码，可以指定语言
```C
int a = 10;
for(int i=0; i<10; i++){
    printf("%d", i);
}
```

# 分割线
分割线1
***
分割线2
---
分割线3
___
当然，三种分割线渲染效果都一样

# 链接语法
这是一个链接 [Markdown语法](https://markdown.com.cn)。

给链接增加Title，
这是一个链接 [Markdown语法](https://markdown.com.cn "最好的markdown教程")。

直接用尖括号    
<https://markdown.com.cn>  
<fake@example.com>


带格式化的链接  
I love supporting the **[EFF](https://eff.org)**.  
This is the *[Markdown Guide](https://www.markdownguide.org)*.  
See the section on [`code`](#code).  

这个链接用 1 作为网址变量 [Google][1]
这个链接用 runoob 作为网址变量 [Runoob][runoob]
然后在文档的结尾为变量赋值（网址）

  [1]: http://www.google.com/
  [runoob]: http://www.runoob.com/

用%20代替空格
[link](https://www.example.com/my%20great%20page)

# 图片
插入图片Markdown语法代码
```
![图片alt](图片链接 "图片title")
```

注意这是本地图片  
![这是图片](./img/philly-magic-garden.jpg "Magic Gardens")

添加链接
[![沙漠中的岩石图片](./img/shiprock.jpg "Shiprock")](https://markdown.com.cn)

# 转义字符
加反斜杠\*, \`，以此类推。

# 内嵌HTML
This **word** is bold. This <em>word</em> is italic.

This is a regular paragraph.

<table>
    <tr>
        <td>Foo</td>
    </tr>
</table>

This is another regular paragraph.