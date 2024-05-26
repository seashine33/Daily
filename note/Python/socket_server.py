"""socket模块使用示例

问题：
    2024.5.7:
        将文件命名为socket.py，此时会报'module' object is not callable的错
        这是由于import socket时，会尝试着导入自身。
"""

import threading
import socket
from colorama import Fore, Back, Style, init

encoding = 'GB2312'  # GB2312, utf-8, gbk
BUFSIZE = 1024

def demo():
    s = socket.socket()         # 创建 socket 对象
    host = socket.gethostname() # 获取本地主机名
    port = 8088                # 设置端口
    s.bind((host, port))        # 绑定端口
    
    s.listen(5)                 # 等待客户端连接
    while True:
        c,addr = s.accept()     # 建立客户端连接
        print('连接地址：', addr)
        c.send('Hello'.encode())
        c.close()      

# 建立TCP连接后，用于接受来自客户端的数据
class Reader(threading.Thread):
    def __init__(self, client):
        threading.Thread.__init__(self)
        self.client = client
        
    def run(self):
        while True:
            data = self.client.recv(BUFSIZE)
            if(data):
                string = bytes.decode(data, encoding)
                print(Fore.RED + string[:-2])
            else:
                break
        print("close:", self.client.getpeername())  # 将客户端传输而来的数据输出为红色

# 建立TCP连接后，用于传输数据给客户端
class Sender(threading.Thread):
    def __init__(self, client):
        threading.Thread.__init__(self)
        self.client = client
        
    def run(self):
        while True:
            send_string = input() + '\r\n'
            if send_string == '+++':
                self.client.close()
            self.client.send(send_string.encode(encoding))

# 用于与客户端建立TCP连接
class Linker(threading.Thread):
    def __init__(self, port):
        threading.Thread.__init__(self)
        self.port = port
        self.sock = socket.socket()
        self.sock.bind(('0.0.0.0', self.port))
        self.sock.listen(5)     # 开始 TCP 监听。 backlog 指定在拒绝连接之前，操作系统可以挂起的最大连接数量。该值至少为 1，大部分应用程序设为 5 就可以了。

    def run(self):
        print("listener started")
        while True:
            client, cltadd = self.sock.accept()     # 被动接受TCP客户端连接,(阻塞式)等待连接的到来
            Reader(client).start()
            Sender(client).start()

if __name__ == '__main__':
    init(autoreset=True)    # 用于配置输出颜色。
    lst = Linker(8088)   # create a listen thread
    lst.start() # then start