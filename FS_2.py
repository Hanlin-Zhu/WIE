#Fucntion 1, section the image into pathces
from __future__ import print_function
import numpy as np
import Image
import sys
import time    

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

def ImportPathch(para,Input):
    Loc[i*batchSize:(i+1)*batchSize]
    Output=np.zeros((numOfSections,para['new_height'],para['new_width']))

def ImportControl(para,a,b,corlabel):
    #for reconstruction
    numOfImgPerCol=(para['height']-para['new_height'])/para['Rstride_height']+1
    numOfImgPerRow=(para['width']-para['new_width'])/para['Rstride_width']+1
    th=(numOfImgPerCol-1)*para['Rstride_height']+para['new_height']
    tw=(numOfImgPerRow-1)*para['Rstride_width']+para['new_width']
    
    Output2=np.zeros((b-a+1,th,tw))
    for i in range(b-a+1):
        pic=np.array(Image.open(str(i+a)+".BMP"))
        pic=pic[:,:,1]/255.
        if corlabel==1:
            mask_=np.random.normal(0,para['cor'],pic.shape)
            pic=pic+mask_
        Output2[i]=pic[0:th,0:tw]

    return Output2

def ImportSec(para,a,b,overlap):
    patchLoc,numSec=ImgSec(para,-1,overlap)
    Output1=np.zeros(((b-a+1)*numSec,5))
    for i in range(b-a+1):
        Output1[i*numSec:(i+1)*numSec],w=ImgSec(para,i,overlap)
    return Output1



def ImgSec(para,pic,overlap):
    if overlap==0:
        s1=para['stride_height']
        s2=para['stride_width']
        para['stride_height']=para['new_height']
        para['stride_width']=para['new_width']
    
    numOfImgPerCol=(para['height']-para['new_height'])/para['stride_height']+1
    numOfImgPerRow=(para['width']-para['new_width'])/para['stride_width']+1

    numOfSections=numOfImgPerRow*numOfImgPerCol;

    # initialize Output Variables as 4D array
    Output=np.zeros((numOfSections,5))
    #Output=np.zeros((numOfSections,para['new_height'],para['new_width'],para['NumColorCh']))
    for i in range(numOfImgPerCol):
        for j in range(numOfImgPerRow):
            Output[i*numOfImgPerRow+j]=[pic,i*para['stride_height'],i*para['stride_height']+para['new_height'],j*para['stride_width'],j*para['stride_width']+para['new_width']]
    
    if overlap==0:
        para['stride_height']=s1
        para['stride_width']=s2
    
    return Output,numOfSections



def ImgComb(para,Input):
    
    numOfImgPerCol=(para['height']-para['new_height'])/para['stride_height']+1
    numOfImgPerRow=(para['width']-para['new_width'])/para['stride_width']+1

    numOfSections=numOfImgPerRow*numOfImgPerCol;

    # initialize Output Variables as 2D array
    Output=np.zeros(((numOfImgPerCol-1)*para['stride_height']+para['new_height'],(numOfImgPerRow-1)*para['stride_width']+para['new_width']))
    OutputAcu=np.zeros(Output.shape)

    for i in range(numOfImgPerCol):
        for j in range(numOfImgPerRow):

            patch=Output[i*para['stride_height']:i*para['stride_height']+para['new_height'],j*para['stride_width']:j*para['stride_width']+para['new_width']];
            patchAcu=OutputAcu[i*para['stride_height']:i*para['stride_height']+para['new_height'],j*para['stride_width']:j*para['stride_width']+para['new_width']];
            patch=patch+Input[i*numOfImgPerRow+j]
            patchAcu=patchAcu+np.ones(Input[i*numOfImgPerRow+j].shape)
            
            
            Output[i*para['stride_height']:i*para['stride_height']+para['new_height'],j*para['stride_width']:j*para['stride_width']+para['new_width']]=patch;
            OutputAcu[i*para['stride_height']:i*para['stride_height']+para['new_height'],j*para['stride_width']:j*para['stride_width']+para['new_width']]=patchAcu;
    
           
    Output=Output/OutputAcu
    return Output



def LayerTrain (para,Loc,Input,Desired_Input,Layer,WeightDict,BiasDict,serverCreate):
    import tensorflow as tf
    import numpy as np
    
    batchSize=para['batchSize'] 
    lam=para['lam']; # this is the parameter multiplying the Frobenius norm (squared) of Weight Matrix, since tied weights are used. we would forget the 1/2 multiplyer
    p_act=para['p_act']; # this is the parameter used in calculating the KL distance of mean activation of hidden layer. 
    l_rate=para['l_rate'];
    n_Hidden=para['n_Hidden'];
    n_Input = para['n_Input']
    n_Output= n_Input
    epoch=para['epoch']
    percentTrain=para['percentTrain']
    FLAGS=para['FLAGS']
    cluster=para['cluster']    
    if serverCreate == 1:
        server = tf.train.Server(cluster,job_name=FLAGS.job_name,task_index=FLAGS.task_index)
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        with tf.device("job:ps/task:0"):
            W_init_max = 4*np.sqrt(6. / (n_Input+n_Hidden)) # A convention used by deep learning society.
            W_init= tf.random_uniform(shape=[n_Input,n_Hidden],
                                      minval=-W_init_max,
                                      maxval=W_init_max)
                
            W = tf.Variable(W_init,name='W')
            b = tf.Variable(tf.zeros([n_Hidden]),name='b')
                                      
            W_prime = tf.transpose (W)
            b_prime = tf.Variable(tf.zeros([n_Output]),name='b_prime')


    # Between-graph replication
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index,cluster=cluster)):
            global_step = tf.get_variable('global_step', [],initializer = tf.constant_initializer(0),trainable = False)
            # add gaussian white noise.
            # One layer of hidden Units currently Assumed, Multi-Layer (Stacked autoencoder) may be added in the future. 
            # The number of hidden units in this layer is a factor multiple of the number of input units. 

            X = tf.placeholder("float",[None,n_Input],name='X')
            XD = tf.placeholder("float",[None,n_Input],name='XD')



            def model(X,W,b,W_prime,b_prime):
        
                Y = tf.nn.sigmoid(tf.matmul(X,W) +b)
                Z = tf.nn.sigmoid(tf.matmul(Y,W_prime)+b_prime)
   
                return Y,Z
            Y,Z = model(X,W,b,W_prime,b_prime);

            HMA= tf.reduce_sum(Y)/n_Hidden #hidden layer's Mean Activation

            cost1 = tf.reduce_sum(tf.pow(XD-Z,2)/2/batchSize) # Mean squared error cost
            cost1m = tf.reduce_mean(tf.pow(XD-Z,2))
            cost2 = tf.reduce_sum(lam*tf.pow(W,2)/2);  # Weight regularization cost
            cost3 =  p_act*tf.log(p_act/HMA)+(1-p_act)*tf.log((1-p_act)/(1-HMA)) # KL distance cost

            cost=cost1+cost2+cost3; 

            train_op = tf.train.GradientDescentOptimizer(l_rate).minimize(cost)

    
        
            init_op = tf.initialize_all_variables()
            print("Variables initialized ...")
                
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),global_step=global_step,init_op=init_op)
        begin_time = time.time()
        frequency = 100
        with sv.prepare_or_wait_for_session(server.target) as sess:
            
                def feedData(Loc,la,lb,RawData,Layer):
                    Output=np.zeros((lb-la,n_Input))
                    for i in range(lb-la):
           # print [Loc[i+la,0],Loc[i+la,1],Loc[i+la,2],Loc[i+la,3],Loc[i+la,4]]
                        patch=RawData[int(Loc[i+la,0]),int(Loc[i+la,1]):int(Loc[i+la,2]),int(Loc[i+la,3]):int(Loc[i+la,4])]
                        Y=patch.reshape(1,patch.shape[0]*patch.shape[1])
                        for j in range (Layer-1):
                            Y = sigmoid(np.matmul(Y,WeightDict[j]) +BiasDict[j])    
                        Output[i]=Y
                    return Output 

                batchTotal=Loc.shape[0]/batchSize;
                print ("batchTotal:"+str(batchTotal))
                print ("epochs:"+str(para['epoch']))
                print ("percentTrain:"+str(para['percentTrain']))#!!!!!!!!!!!!!! THIS PARAMETER ALSO HEAVILY CONTROLS THE SINGLE BATCH TRAINING TIME.


                TestIn=feedData(Loc,int(batchTotal*percentTrain)*batchSize,Loc.shape[0],Input,Layer)
                TestDesired=feedData(Loc,int(batchTotal*percentTrain)*batchSize,Loc.shape[0],Desired_Input,Layer)                              
                for j in range(epoch):   # Do this training for x epochs 
        
                    totalIter= epoch*int(batchTotal*percentTrain)
                    for i in range(int(batchTotal*percentTrain)):   # batchSize =, 80% batches as training set, the last 20% batches as validation set,

                        sess.run(train_op, feed_dict=({X:feedData(Loc,i*batchSize,(i+1)*batchSize,Input,Layer),XD:feedData(Loc,i*batchSize,(i+1)*batchSize,Desired_Input,Layer)}))
            
                        perform=sess.run(cost1m, feed_dict=({X:TestIn,XD:TestDesired}))
                        progress=j*int(batchTotal*percentTrain)+i
                        print(str(perform)+"["+str(progress+1)+"/"+str(totalIter)+"]")

                W_final=sess.run(W)
                b_final=sess.run(b)
                bp_final=sess.run(b_prime)
        sv.stop()
        print("done")



    return W_final,b_final,bp_final


def JointTrain (para,Loc,Input,Desired_Input,Layer,WeightDict_I,BiasDict_I,BiasPrimeDict_I):
    import tensorflow as tf
    import numpy as np

    batchSize=para['batchSize'] 
    lam=para['lam']; # this is the parameter multiplying the Frobenius norm (squared) of Weight Matrix, since tied weights are used. we would forget the 1/2 multiplyer
    p_act=para['p_act']; # this is the parameter used in calculating the KL distance of mean activation of hidden layer. 
    l_rate=para['l_rate'];
    n_Hidden=para['n_Hidden'];
    n_Input = para['n_Input']
    n_Output= n_Input
    epoch=para['epoch']
    percentTrain=para['percentTrain']
  # add gaussian white noise.
# One layer of hidden Units currently Assumed, Multi-Layer (Stacked autoencoder) may be added in the future. 
# The number of hidden units in this layer is a factor multiple of the number of input units. 

    LayerSize=np.zeros(Layer+1)
    for i in range(Layer):
        LayerSize[i]=WeightDict_I[i].shape[0] 
    LayerSize[Layer]=WeightDict_I[Layer-1].shape[1]
    print (LayerSize)


    X = tf.placeholder("float",[None,LayerSize[0]],name='X')
    XD = tf.placeholder("float",[None,LayerSize[0]],name='XD')

    WeightDict={'Name':'Weight_Dict'}   #Name doesn't matter actually, Just to initialize the Dictionary with at least one variable,
# I honestly don't know if I can intialize with zero variables (empty dictionary)
    WeightPrimeDict={'Name':'WeightPrimeDict'}
    BiasDict={'Name':'Bias_Dict'}
    BiasPrimeDict={'Name':'Bias_Prime_Dict'}

    for i in range(LayerSize.shape[0]-1):
    
        WeightDict[i]=tf.Variable(WeightDict_I[i],name='W_'+str(i))
        BiasDict[i]=tf.Variable(BiasDict_I[i],name='b_'+str(i))
        BiasPrimeDict[i]=tf.Variable(BiasPrimeDict_I[i],name='bP_'+str(i))
        WeightPrimeDict[i]=tf.transpose (WeightDict[i])
    
    
    def model_Joint(X,WeightDict,BiasDict,WeightPrimeDict,BiasPrimeDict):
    #Y=X
        for i in range (Layer):
            X = tf.nn.sigmoid(tf.matmul(X,WeightDict[i]) +BiasDict[i])
    #Z=Y
        for i in range (Layer):
            X = tf.nn.sigmoid(tf.matmul(X,WeightPrimeDict[Layer-1-i]) +BiasPrimeDict[Layer-1-i])
        return X

    Z = model_Joint(X,WeightDict,BiasDict,WeightPrimeDict,BiasPrimeDict);
#________________________________________________________________________________New code above this line

    cost1 = tf.reduce_sum(tf.pow(XD-Z,2)/2/batchSize) # Mean squared error cost
    cost1m = tf.reduce_mean(tf.pow(XD-Z,2))
    cost2 = 0
    for i in range(Layer):
        cost2=cost2+lam*tf.reduce_sum(tf.pow(WeightDict[i],2)/2);  # Weight regularization cost
    cost=cost1+cost2

    train_op = tf.train.GradientDescentOptimizer(l_rate).minimize(cost)

    sess = tf.Session()

    init = tf.initialize_all_variables() 

    sess.run(init)

    def feedData(Loc,la,lb,RawData):
        Output=np.zeros((lb-la,LayerSize[0]))
        for i in range(lb-la):
           # print [Loc[i+la,0],Loc[i+la,1],Loc[i+la,2],Loc[i+la,3],Loc[i+la,4]]
            patch=RawData[int(Loc[i+la,0]),int(Loc[i+la,1]):int(Loc[i+la,2]),int(Loc[i+la,3]):int(Loc[i+la,4])]
            Y=patch.reshape(1,patch.shape[0]*patch.shape[1])
            Output[i]=Y
        return Output 

    batchTotal=Loc.shape[0]/batchSize;
    print ("batchTotal:"+str(batchTotal))
    print ("epochs:"+str(para['epoch']))
    print ("percentTrain:"+str(para['percentTrain']))#!!!!!!!!!!!!!! THIS PARAMETER ALSO HEAVILY CONTROLS THE SINGLE BATCH TRAINING TIME.


    TestIn=feedData(Loc,int(batchTotal*percentTrain)*batchSize,Loc.shape[0],Input)
    TestDesired=feedData(Loc,int(batchTotal*percentTrain)*batchSize,Loc.shape[0],Desired_Input)                              
    for j in range(epoch):   # Do this training for x epochs 
        
        totalIter= epoch*int(batchTotal*percentTrain)
        for i in range(int(batchTotal*percentTrain)):   # batchSize =, 80% batches as training set, the last 20% batches as validation set,

            sess.run(train_op, feed_dict=({X:feedData(Loc,i*batchSize,(i+1)*batchSize,Input),XD:feedData(Loc,i*batchSize,(i+1)*batchSize,Desired_Input)}))
            
            perform=sess.run(cost1m, feed_dict=({X:TestIn,XD:TestDesired}))
            progress=j*int(batchTotal*percentTrain)+i
            print (str(perform)+"["+str(progress+1)+"/"+str(totalIter)+"]")
    W_final=WeightDict_I
    b_final=BiasDict_I
    bp_final=BiasPrimeDict_I
    for i in range(Layer):     
        W_final[i]=sess.run(WeightDict[i])
        b_final[i]=sess.run(BiasDict[i])
        bp_final[i]=sess.run(BiasPrimeDict[i])

    sess.close()
    return W_final,b_final,bp_final
