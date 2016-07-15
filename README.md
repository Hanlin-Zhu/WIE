# Denoising Auto Encoder 
# Distributed Tensorflow Example on Google Cloud Platform (GCE)

Using data parallelism with shared model parameters while updating parameters asynchronous.

Trains a simple denoising autoencoder for 2 epochs on three machines using one parameter server.

目前多层hidden layers layer-wise pre-training用来initialize weights 然后所有layers joint training的模式只在单机版上实现了。
在这个Distributed DAE的例子中只使用了一个hidden layer. t2.py中的71-82行因此被comment掉了。时间原因没能解决的问题是： 之前单机版对应的t2.py 在68行运行完成FS_2.LayerTrain（）后该函数中所定义的所有参数（或者说整个Graph)都被抹掉了，所以第76行再次调用 FS_2.LayerTrain（）时就可以根据新的para要求重新建立所有参数。但在分布式运行中，运行完68行，好像原来的参数（其实是整个Graph)不会消失。再次调用FS_2.LayerTrain时76行，会报错，基本是说参数已经存在了，无法修改。所以最简单粗暴的解决方案是一次性把两个layer的参数都造出来，而不是通过调用两次FS_2.LayerTrain（）重复创建同名但是形状不同（第二个layer的神经元比第一个layer多n倍，见t2.py 73行）的参数。

A Simple Google Cloud Platform version is detailed as the following 

Step 1. Register for Google Cloud Platform ( you get 60 days trials and $300 for free)  https://cloud.google.com/

Step 2. Go to your console (main page of GCE) and  create a compute machine instance. 
        a. click the 3 horizontal bars icon on the top left corner, it gives a list of services and products 
        b. select compute->compute engine->create an instance
        c. use the name pc-1 and other default settings like 1vCPU, 3.75G, enable HTTP and HTTPS traffic
        
Step 3. Once pc-1 has been created. you would repeat Step2 and create pc-2, pc-3 and pc-4,A list of compute engines created should be shown together with their internal and external IPs.

Step 4. click on the SSH connect icon in the pc-1 row to access the first virtual machine. 

Step 5. Install git, python, pip, tensorflow, on pc-1 by performing the following commands:
```
   sudo apt-get update
   sudo apt-get install git 
   sudo apt-get install python-dev python-pip python-imaging
   export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0-cp27-none-linux_x86_64.whl
   sudo pip install --upgrade $TF_BINARY_URL
   
```
Step 6. clone this directory 
```
   git clone https://github.com/Hanlin-Zhu/WIE.git
```
这一步主要是在每一台机器上加载代码和训练数据集（下载WIE这个folder)，其实不一定用github, 我只是觉得这样比较方便。
Step 7. Repeat Step 4 and 6 for all the other three virtual machines

Step 8. Access each machine through SSH connect and run the corresponding example.py, the program won't start and may either freeze or raise connection failed error until you have run the program on all machines. 
```
pc-01$ cd WIE
pc-01$ python t2.py --job_name="ps" --task_index=0 

pc-02$ cd WIE
pc-02$ python t2.py --job_name="worker" --task_index=0 

pc-03$ cd WIE
pc-03$ python t2.py --job_name="worker" --task_index=1 

pc-04$ cd WIE
pc-04$ python t2.py --job_name="worker" --task_index=2 

```
注意： 

1. 在把整张图片切小片(patch)用来训练的过程中，其实并没有产生很多小图片（为了节省内存，尤其是小图片之间有overlap时，保存所有小图片所占用空间会比原始图片大很多）。我只是把每一张小图片上在原始图片上的起止坐标保存在了
Loc_overlap参数中,以后根据这个参数再切出每个训练batch所需的训练集。


2. 分布式运算的核心是把需要训练的一套parameters (FS_2.py line 118) 指定在一台机器（job name:ps) 上。然后把所有训练的动作的 (FS_2.py line 132)在指定在三台其他机器上（job name: worker)上。 在这个程序中，三台worker（彼此之前通过task_index区分）都读取了所有的training data（12batch) 2 遍（2 epochs)好像并没有降低总时间。但实际上可以理解为在一个单机版的程序中训练了6遍(epochs)， 降低的实际上是训练一个epoch(定义是pass through了所有training data 一遍）的时间。从训练效果（两张截图的对比）可以看出。 

3. google cloud 在不用的时候可以通过console界面右上角第一个键 Google cloud shell 命令行中通过
``` gcloud compute instances stop pc-1 pc-2 pc-3 pc-4```来暂时休眠以减少扣费。（不会自动清除数据）
下次在使用时只需要通过
``` gcloud compute instances start pc-1 pc-2 pc-3 pc-4```来唤醒就可以。

4. 测试的结果会保存在result文件夹中。 G,C,R字母分别代表good(原始训练图片）,corrupted(加入了噪音的图片），reconstructed （把corrupted 的图片切小块以后通过DAE，然后在在每个去噪后的小片拼起来，小片之间overlap的部分对pixel intensity取平均）
主要的参考：



1.	官方文档
https://www.tensorflow.org/versions/r0.9/how_tos/distributed/index.html#replicated-training

2.	Imanol Schlag的simple mnist 在github 上的例子
 https://github.com/ischlag/distributed-tensorflow-example

3.     Xie, J., Xu, L., & Chen, E. (2012). Image denoising and inpainting with deep neural networks. In Advances in Neural Information Processing Systems (pp. 341-349).

4.      Nlintz 的autoencoder tutorial on github
https://github.com/nlintz/TensorFlow-Tutorials/blob/master/06_autoencoder.py
