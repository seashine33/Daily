# 饥荒服务器
- ip
  - master: 120.55.69.84
  - cave: 121.43.105.129
- 教程
  - [事无巨细的Steam饥荒联机云服搭建教程](https://blog.csdn.net/Hydius/article/details/106121931)
  - [SteamCMD开发者社区](https://developer.valvesoftware.com/wiki/SteamCMD)
  - [Steam版饥荒双专用服务器搭建教程](https://blog.csdn.net/XUdashi/article/details/118695063)
- 流程
  - root用户
    - adduser steam
    - usermod -aG sudo steam
    - su steam
  - steam用户
    - sudo apt-get update
    - sudo apt-get upgrade
    - sudo apt-get install lib32gcc-s1
    - sudo add-apt-repository multiverse
    - sudo dpkg --add-architecture i386
    - sudo apt install libstdc++6 libgcc1 libcurl4-gnutls-dev:i386 lib32z1
      - cd /usr/lib/
      - sudo ln -s libcurl.so.4 libcurl-gnutls.so.4
    - 报libstdc++.so.6的错
      - sudo apt-get install libstdc++6 
      - sudo apt-get install lib32stdc++6
    - sudo apt-get install screen
    - sudo apt install unzip
    - sudo apt update
    - mkdir steamcmd
    - cd steamcmd
    - wget -P ~/steamcmd https://steamcdn-a.akamaihd.net/client/installer/steamcmd_linux.tar.gz
    - tar -xvzf steamcmd_linux.tar.gz
    - ./steamcmd.sh
      - force_install_dir /home/qh/dst
      - login anonymous
      - app_update 343050 validate
      - exit
  - klei开服
    - https://accounts.klei.com/login
    - 获取token
  - 存档
    - 服务器标签写入cluster_token.txt
- 启动
  - ./dontstarve_dedicated_server_nullrenderer -console -cluster "Cluster_1"
  - ./dontstarve_dedicated_server_nullrenderer -console -cluster "Cluster_1" -shard Master
  - ./dontstarve_dedicated_server_nullrenderer -console -cluster "Cluster_1" -shard Caves
- 双服务器
  - 开端口不是命令行开，是云服务器控制台的安全组里面开，协议类型是UDP。
- 添加mod
  - 服务器目录mods文件夹下/dedicated_server_mods_setup.lua
- 本地存档
  - C:\Users\69453\Documents\Klei\DoNotStarveTogether\485678640
- 本地mod
  - E:\SteamLibrary\steamapps\common\Don't Starve Together\mods



--防卡两招
ServerModSetup("1216718131")
ServerModCollectionSetup("1216718131")

--Extra Equip Slots
ServerModSetup("375850593")
ServerModCollectionSetup("375850593")

--Always fresh
ServerModSetup("462372013")
ServerModCollectionSetup("462372013")

--快速采集
ServerModSetup("501385076")
ServerModCollectionSetup("501385076")

--999堆叠
ServerModSetup("831523966")
ServerModCollectionSetup("831523966")

--Global Positions
ServerModSetup("378160973")
ServerModCollectionSetup("378160973")

--复活按钮和传送按钮
ServerModSetup("2753774601")
ServerModCollectionSetup("2753774601")

--简易血条DST
ServerModSetup("1207269058")
ServerModCollectionSetup("1207269058")

--Fast Travel (GUI)
ServerModSetup("1530801499")
ServerModCollectionSetup("1530801499")