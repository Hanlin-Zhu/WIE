# testControl.py
# import test
import numpy as np
import FS_2
import Image
import webbrowser as wb
import time
import os
import tensorflow as tf
parameter_servers = ["pc-1:2222"]
workers = [	"pc-2:2222", 
			"pc-3:2222",
			"pc-4:2222"]
cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

para={'Name':'cluster'}
para={'cluster':'cluster'}
para['TrainSample']=[1,2]# use 1.bmp to 5.bmp as training images
para['TestSample']=[3,4]# use 6.bmp to 8.bmp as test images
para['height']=700
para['width']=700
para['NumColorCh']=1  # RGB image assumed. but turned to grayscale in this program
# para['fileType']='PNG'
para['new_height']=70
para['new_width']=70

para['stride_height']=70
para['stride_width']=70

para['Rstride_height']=70
para['Rstride_width']=70
para['LayerSize']=2 #number of Hidden Layers
para['percentTrain']=0.85 #!!!!!!!!!!!!!! THIS PARAMETER ALSO HEAVILY CONTROLS THE SINGLE BATCH TRAINING TIME.
para['cor']=0.1
para['batchSize']=50
para['lam']=0; # this is the parameter multiplying the Frobenius norm (squared) of Weight Matrix, since tied weights are used. we would forget about the 1/2 multiplyer
para['p_act']=0.95; # this is the parameter used in calculating the KL distance of mean activation of hidden layer. 
para['l_rate']=0.01;
para['epoch']=2; 
para['n_Input'] = para['new_height']*para['new_width']
para['Hidden_Input_Fac']=2
para['n_Hidden']= para['n_Input']*para['Hidden_Input_Fac']

tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS
para['FLAGS']=FLAGS

Input=FS_2.ImportControl(para,para['TrainSample'][0],para['TrainSample'][1],1)
Desired_Input=FS_2.ImportControl(para,para['TrainSample'][0],para['TrainSample'][1],0)
Loc_overlap=FS_2.ImportSec(para,para['TrainSample'][0],para['TrainSample'][1],1)
Loc=FS_2.ImportSec(para,para['TrainSample'][0],para['TrainSample'][1],0)

np.random.seed(1)
np.random.shuffle(Loc_overlap)

# One layer of hidden Units currently Assumed, Multi-Layer (Stacked autoencoder) may be added in the future. 
# The number of hidden units in this layer is a factor multiple of the number of input units. 
#noisyX=mask+X;

WeightDict={'Name':'Weight_Dict'}   #Name doesn't matter actually, Just to initialize the Dictionary with at least one variable,
# I honestly don't know if I can intialize with zero variables (empty dictionary)
BiasDict={'Name':'Bias_Dict'}
BiasPrimeDict={'Name':'Bias_Prime_Dict'}

#-------------------------------  Start Training  -------------------------------------
Layer=1;
WeightDict[0],BiasDict[0],BiasPrimeDict[0]=FS_2.LayerTrain(para,Loc_overlap,Input,Desired_Input,Layer,WeightDict,BiasDict)


#------------------------------- First Layer Trained -------------------------------------
para['n_Input']=para['n_Hidden']
para['n_Hidden']=para['n_Hidden']*para['Hidden_Input_Fac']

Layer=2;
WeightDict[1],BiasDict[1],BiasPrimeDict[1]=FS_2.LayerTrain(para,Loc_overlap,Input,Desired_Input,Layer,WeightDict,BiasDict)

#------------------------------- Second Layer Trained -------------------------------------
#para['l_rate']=0.0001;
#para['lam']=0;
#WeightDict,BiasDict,BiasPrimeDict=FS_2.JointTrain(para,Loc,Input,Desired_Input,Layer,WeightDict,BiasDict,BiasPrimeDict)
#------------------------------- Joint Training Finished -------------------------------------

#_________________________________Assume that all training are finished,How do we evaluate___________


# N hidden layers in total   only supports 2 layers now

def model_Joint(X,LayerSize,WeightDict,BiasDict,BiasPrimeDict):
    Y=X
    for i in range (LayerSize):
        Y = FS_2.sigmoid(np.matmul(Y,WeightDict[i]) +BiasDict[i])
    Z=Y
    for i in range (LayerSize):
        Z = FS_2.sigmoid(np.matmul(Z,np.transpose(WeightDict[LayerSize-1-i])) +BiasPrimeDict[LayerSize-1-i])
    return Z


#___________________________________________________________________
Input=FS_2.ImportControl(para,para['TestSample'][0],para['TestSample'][1],1)
Desired_Input=FS_2.ImportControl(para,para['TestSample'][0],para['TestSample'][1],0)


Recon=np.zeros((Input.shape))
ReconAcu=np.zeros(Recon.shape)
numOfImgPerCol=(para['height']-para['new_height'])/para['Rstride_height']+1
numOfImgPerRow=(para['width']-para['new_width'])/para['Rstride_width']+1
print ("start reconstructing...")
for m in range(Input.shape[0]):#scan over all images.
    for i in range(numOfImgPerCol):
        print ("["+str(m*numOfImgPerCol*numOfImgPerRow+i*numOfImgPerRow)+"/"+str(Input.shape[0]*numOfImgPerCol*numOfImgPerRow)+"]")
        for j in range(numOfImgPerRow):  
            
            patchOri=Input[m,i*para['Rstride_height']:i*para['Rstride_height']+para['new_height'],j*para['Rstride_width']:j*para['Rstride_width']+para['new_width']]
            
            temp=patchOri.reshape(1,patchOri.shape[0]*patchOri.shape[1])
            
            patchNew=model_Joint(temp,para['LayerSize'],WeightDict,BiasDict,BiasPrimeDict)# Key line here
            
            patchNew=patchNew.reshape(para['new_height'],para['new_width'])
            

            patch=Recon[m,i*para['Rstride_height']:i*para['Rstride_height']+para['new_height'],j*para['Rstride_width']:j*para['Rstride_width']+para['new_width']];
            patchAcu=ReconAcu[m,i*para['Rstride_height']:i*para['Rstride_height']+para['new_height'],j*para['Rstride_width']:j*para['Rstride_width']+para['new_width']];
            
            patch=patch+patchNew
            patchAcu=patchAcu+np.ones(patchNew.shape)
            
            
            Recon[m,i*para['Rstride_height']:i*para['Rstride_height']+para['new_height'],j*para['Rstride_width']:j*para['Rstride_width']+para['new_width']]=patch;
            ReconAcu[m,i*para['Rstride_height']:i*para['Rstride_height']+para['new_height'],j*para['Rstride_width']:j*para['Rstride_width']+para['new_width']]=patchAcu;
  
Recon[m]=Recon[m]/ReconAcu[m]
#RECONSTRUCTION COMPLETE__________________________________________

PSNR_Cor=10*np.log10((255.**2)/(255.**2)/np.mean(np.power((Desired_Input-Input),2)))
print(PSNR_Cor)
PSNR_Rec=10*np.log10((255.**2)/(255.**2)/np.mean(np.power((Desired_Input-Recon),2)))
print(PSNR_Rec)

#PSNR REPORT COMPLETE__________________________________________
savePath="/result/"+time.strftime("20%y-%m-%d--%H:%M:%S")
os.mkdir(savePath)
fileType="BMP"
for j in range(Input.shape[0]): 
    
    pil_im0 = Image.fromarray(np.uint8(Desired_Input[j]*255)) #corrupted image
    pil_im0.save(savePath+"/"+str(j+para['TestSample'][0])+"G."+fileType,fileType) 

    pil_im1 = Image.fromarray(np.uint8(Input[j]*255)) #corrupted image
    pil_im1.save(savePath+"/"+str(j+para['TestSample'][0])+"C."+fileType,fileType) 

    pil_im2 = Image.fromarray(np.uint8(Recon[j]*255)) #corrupted image
    pil_im2.save(savePath+"/"+str(j+para['TestSample'][0])+"R."+fileType,fileType) 
f=open(savePath+"/para.py",'w')
f.write(repr(para))
f.close()

#SAVING COMPLETE________________________________________________

#Joint Training TO BE CONTINUED_________________________________

#np.savetxt('W1.pyData',W,delimiter=',')
#np.savetxt('b1.pyData',b,delimiter=',')
#np.savetxt('bp1.pyData',bp,delimiter=',')
#np.savetxt('W.pyData',WeightDict,delimiter=',')
#np.savetxt('b.pyData',b,delimiter=',')
#np.savetxt('bp.pyData',bp,delimiter=',')

#wb.open("2.PNG") # display corrupted image patches

#W1=np.genfromtxt('W.pyData',delimiter=',')
#b1=np.genfromtxt('b.pyData',delimiter=',')
#bp1=np.genfromtxt('bp.pyData',delimiter=',')

#W2=np.genfromtxt('W1.pyData',delimiter=',')
#b2=np.genfromtxt('b1.pyData',delimiter=',')
#bp2=np.genfromtxt('bp1.pyData',delimiter=',')

#    for i in range (LayerSize):
#      np.savetxt('W'+str(i)+'.pyData',WeightDict,delimiter=',')
#      np.savetxt('B'+str(i)+'.pyData',BiasDict,delimiter=',')
 #     np.savetxt('BP'+str(i)+'.pyData',BiasPrimeDict,delimiter=',')


