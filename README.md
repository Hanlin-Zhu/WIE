# Denoising Auto Encoder 
# Distributed Tensorflow Example on Google Cloud Platform (GCE)

Using data parallelism with shared model parameters while updating parameters asynchronous. See comment for some changes to make the parameter updates synchronous 

Trains a simple denoising autoencoder for 2 epochs on three machines using one parameter server.

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

主要的参考：

1.	官方文档
https://www.tensorflow.org/versions/r0.9/how_tos/distributed/index.html#replicated-training

2.	Imanol Schlag的simple mnist 在github 上的例子
 https://github.com/ischlag/distributed-tensorflow-example



