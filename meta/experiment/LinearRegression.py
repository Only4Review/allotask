import numpy as np

# fix data size and distribution

# number od data points for meta-training tasks, train and validation
ntr=10
nts=2
# number od data points for meta-testing tasks, adaptation and test
nmtstr=20
nmtsts=50
# number of input components ("p" in text)
d=128
# number of tasks for meta-training ("m" in text)
T=3
# number of tasks for meta-testing
Tmts=100
# inner learning rate for meta-training
alptr=0.3
# inner learning rate for meta-testing
alpts=0.3

# output noise
sigma=0.2

# mean task 
#Wmean=np.random.normal(0.,1./np.sqrt(d),[d,1])
Wmean=0.05*np.ones([d,1])
#Wmean=np.zeros([d,1])

# task variability ("nu" in text)
sigmatask=0.2

# number of repetitions
nrep=10

Lrec=np.empty([nrep,])
Lrect=np.empty([nrep,])


for irep in range(nrep):
        
    # sample tasks
    # meta-training
    W=np.random.normal(Wmean,sigmatask/np.sqrt(d),[d,T])
    # meta-testing
    Wmts=np.random.normal(Wmean,sigmatask/np.sqrt(d),[d,Tmts])

    # sample data
    # sample input data for meta-training (training and validation)
    Xtr=np.random.normal(0.,1.,[ntr,T,d])
    Xts=np.random.normal(0.,1.,[nts,T,d])
    # sample input data for meta-testing (adaptation and testing)
    Xmtstr=np.random.normal(0.,1.,[nmtstr,Tmts,d])
    Xmtsts=np.random.normal(0.,1.,[nmtsts,Tmts,d])
    # sample output noise for meta-training, training and validation
    Ztr=np.random.normal(0.,sigma,[ntr,T])
    Zts=np.random.normal(0.,sigma,[nts,T])
    # sample output noise for meta-testing, adaptation and testing
    Zmtstr=np.random.normal(0.,sigma,[nmtstr,Tmts])
    Zmtsts=np.random.normal(0.,sigma,[nmtsts,Tmts])
    # compute output data for meta-training (training and validation)
    Ytr=np.empty([ntr,T])
    Yts=np.empty([nts,T])
    for t in range(T):
        Ytr[:,t]=Xtr[:,t,:]@W[:,t]
        Yts[:,t]=Xts[:,t,:]@W[:,t]
    Ytr+=Ztr
    Yts+=Zts
    # compute output data for meta-testing (adaptation and testing)
    Ymtstr=np.empty([nmtstr,Tmts])
    Ymtsts=np.empty([nmtsts,Tmts])
    for t in range(Tmts):
        Ymtstr[:,t]=Xmtstr[:,t,:]@Wmts[:,t]
        Ymtsts[:,t]=Xmtsts[:,t,:]@Wmts[:,t]
    Ymtstr+=Zmtstr
    Ymtsts+=Zmtsts

    # hyperparameters for meta-training inner loop
    # inner learning rate (adjusted according to ntr)
    alpha=alptr/ntr
    # number of update steps in inner loop
    K=1

    # inner loop, full batch, separate tasks
    I=np.identity(d)
    A=np.empty([d,d,T])
    b=np.zeros([d,T])
    for t in range(T):
        A[:,:,t]=I
        for k in range(K):
            b[:,t]+=alpha*A[:,:,t]@Xtr[:,t,:].T@Ytr[:,t]
            A[:,:,t]=(I-alpha*Xtr[:,t,:].T@Xtr[:,t,:])@A[:,:,t]

    # outer loop
    AA=np.zeros([nts*T,d])
    bb=np.zeros([nts*T,])
    for t in range(T):
        AA[t*nts:(t+1)*nts,:]=Xts[:,t,:]@A[:,:,t]
        bb[t*nts:(t+1)*nts,]=Yts[:,t]-Xts[:,t,:]@b[:,t]
    w_hat=np.linalg.pinv(AA)@bb

    # adaptation to meta-training data and compute meta-training loss
    W_adapt=np.zeros([d,T])
    L=0
    for t in range(T):
        W_adapt[:,t]=A[:,:,t]@w_hat+b[:,t]
        L+=np.linalg.norm(Yts[:,t]-Xts[:,t,:]@W_adapt[:,t])**2
    L=L/(2*nts*T)

    # hyperparameters for meta-testing inner loop
    # inner learning rate (adjusted according to ntr)
    alpha=alpts/nmtstr
    # number of update steps in inner loop
    K=1

    # adaptation to meta-testing data and compute meta-testing loss
    I=np.identity(d)
    Amts=np.empty([d,d,Tmts])
    bmts=np.zeros([d,Tmts])
    for t in range(Tmts):
        Amts[:,:,t]=I
        # k should be in descending order, when data depends on k
        for k in range(K):
            bmts[:,t]+=alpha*Amts[:,:,t]@Xmtstr[:,t,:].T@Ymtstr[:,t]
            Amts[:,:,t]=(I-alpha*Xmtstr[:,t,:].T@Xmtstr[:,t,:])@Amts[:,:,t]

    Wmts_adapt=np.zeros([d,Tmts])

    Lmts=0
    for t in range(Tmts):
        Wmts_adapt[:,t]=Amts[:,:,t]@w_hat+bmts[:,t]
        Lmts+=np.linalg.norm(Ymtsts[:,t]-Xmtsts[:,t,:]@Wmts_adapt[:,t])**2

    Lmts=Lmts/(2*nmtsts*Tmts)

    # record loss

    Lrec[irep]=Lmts
    Lrect[irep]=L

np.savez('LinearRegression.npz',Lrec=Lrec,Lrect=Lrect,d=d,nrep=nrep,sigma=sigma,sigmatask=sigmatask,alptr=alptr,alpts=alpts,ntr=ntr,nts=nts,nmtstr=nmtstr,nmtsts=nmtsts,T=T,Tmts=Tmts,Wmean=Wmean)


