---
layout: post
title: win10安装CUDA和cuDNN
date: 2020-07-08
categories: 深度学习
tags: CUDA
---

### 1.下载CUDA和cuDNN

#### [CUDA下载网址](https://developer.nvidia.com/cuda-downloads)
选择要下载的类型
![](/images/posts/2020/07/0801.png)

#### [cuDNN下载网址](https://developer.nvidia.com/rdp/cudnn-download)
cuDNN下载需要英伟达开发者账户，注册一个就行了。

### 2.安装CUDA和cuDNN
CUDA无脑装就行了，默认装到C盘。
cuDNN是一个压缩包，直接解压。

找到CUDA的安装目录，一般为`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v版本`，找到bin、include、lib目录，将cuDNN压缩包内对应的文件复制到bin、include、lib目录。

**注：是复制bin、include、lib目录中的所有文件到CUDA的相应目录，不是复制bin、include、lib文件夹**

### 3.添加环境变量

![](/images/posts/2020/07/0802.png)
右键我的电脑，依次点击，打开环境变量。

![](/images/posts/2020/07/0803.png)
需要新建两个变量，照图中操作，找到path，点击编辑，再点击新建，会出现下图情况

![](/images/posts/2020/07/0804.png)
然后点击浏览，添加路径
`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2`

和路径

`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\lib\x64`

注：是你的安装路径

### 4.检查是否安装成功

cmd进入`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\extras\demo_suite`

输入`.\bandwidthTest.ext`

出现`result=PASS`，则安装成功

![](/images/posts/2020/07/0805.png)




