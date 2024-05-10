"""socket模块使用示例

问题：
    2024.5.7:
        将文件命名为socket.py，此时会报'module' object is not callable的错
        这是由于import socket时，会尝试着导入自身。
"""

import threading
import socket

encoding = 'utf-8'
BUFSIZE = 1024

class Reader(threading.Thread):
    def __init__(self, client):
        threading.Thread.__init__(self)
        self.client = client
        
    def run(self):
        while True:
            data = self.client.recv(BUFSIZE)
            if(data):
                string = bytes.decode(data, encoding)
                print(string)
            else:
                break
        print("close:", self.client.getpeername())
        
class Sender(threading.Thread):
    def __init__(self, client):
        threading.Thread.__init__(self)
        self.client = client
        
    def run(self):
        while True:
            send_string = input("传输给客户端的内容：\n")
            if send_string == '+++':
                self.client.close()
            self.client.send(send_string.encode())

def demo():
    s = socket.socket()         # 创建 socket 对象
    host = socket.gethostname() # 获取本地主机名
    port = 8088                # 设置端口号
    
    s.connect((host, port))
    
    print(s.recv(1024).decode())
    s.close()


class Client(threading.Thread):
    def __init__(self, port):
        threading.Thread.__init__(self)
        self.port = port
        self.sock = socket.socket()

    def run(self):
        print("connect started")
        host = socket.gethostname()
        self.sock.connect((host, self.port))
        while True:
            Reader(self.sock).start()
            Sender(self.sock).start()


if __name__ == '__main__':
    c = Client(8088)
    c.start()