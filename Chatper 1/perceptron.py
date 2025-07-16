from  random import random
import numpy as np

#activation function that will be used
def sign(x):
    if x>=0:
        sign=1
    else:
        sign=-1
    return sign
#We will assume that data ara ndarray
#(x_val, y_val) will be used when early_stopping=True
class perceptron:
    def __init__(self, x_train, y_train, x_val=None, y_val=None,
                  w=None, bias=True, rand_w_init=True, early_stopping=False, learning_rate=1):
        #...
        if type(x_train).__name__!='ndarray' or x_train.ndim!=2: #it also has to be done and x _val
            pass
        if len(y_train)!=x_train.shape[0]   :#it also has to be done for x_val
            pass
        if w!=None and type(x_train).__name__!='ndarray' and w.ndim!=1 and  w.shape[0]!=x_train.shape[1]:
            pass
        if w!=None and rand_w_init==True:
            #it will be managed as a warning, saying that w will be used
            pass
        if w==None and rand_w_init==False:
            #it will be managed as a warning, saying that random weights will be used
            pass
        #...ERRORS THAT WILL BE DISCUSSED (AS ATTRIBUTES CONVALIDATION) AT THE END
        if bias:
            #add 1 as last (after also for weight) element of each row in x_train and x_val
            x_train = np.column_stack([x_train, np.ones(x_train.shape[0])])
            if x_val!=None:
                x_val = np.column_stack([x_val, np.ones(x_val.shape[0])])
        self.x_train=x_train
        self.x_val=x_val
        self.y_train=y_train
        self.y_val=y_val
        self.n_sample_train=self.x_train.shape[0]
        self.n_features=self.x_train.shape[1]
        if w!=None:
            if bias:
                self.w=np.append(w,1)
            else: 
                self.w=w
        if rand_w_init and w==None:
            w=[random()*2-1 for i in range(self.n_features)]
            #if rand_w_init is initialized as True the components of w_0 will be in [-1,1]
            if bias:
                w.append(random()*2-1)
                #last element of the list is the bias
            self.w=np.array(w)
        self.w_epoch=np.array([self.w])#there will be all the weights at the end of each epoch and is initialized as 
                                       #the matrix having w_0 as the first and only row (next weights will be row to append) 
        if self.x_val!=None:
            self.error_val=[np.inf]#there will be the error on the valid set at each epoch (the first element is necessary 
                                    #for the early stopping)
        self.error_train=[self.test_sample(self.x_train, self.y_train)]#there will be the error on the training set at each epoch
        self.early_stopping=early_stopping
        self.learning_rate=learning_rate
    def train(self, epoch):
        #in this perceptron batch_size=1, so each epoch corresponds to having seen n example, where n=x_train.shape[0]
        #IMPORTANT: at each epoch it's not garanteed that each example of the train will be drawn, but probably some 
                    #example will be drawn multiple times
        to_be_perm=np.arange(0, self.n_sample_train)#array that will be permuted at each epoch, such that each example of
                                                #the training set will be seen once in each epoch
        for i in range(epoch):
            perm=np.random.permutation(to_be_perm)#order on how i-epoch see the examples of the training set
            for j in perm:
                error_example=self.predict_one(self.x_train[j])-self.y_train[j]
                if error_example!=0:
                    self.update_weight(j)
            self.w_epoch=np.append(self.w_epoch, self.w, axis=0)
            error_train_epoch=self.test_sample(self.x_train,self.y_train)#get error on train at the end of epoch...
            self.error_train.append[error_train_epoch]#...and store it
            error_val_epoch=self.test_sample(self.x_val,self.y_val)
            self.error_val.append(error_val_epoch)
            #EARLY STOPPING: training will be stopped when at the end of an epoch the error on the validation set is 
            #greater than the error at the end of the previously epoch (the weight trained at the end of
            #previously epoch will be selected, saved in w_epoch). Error will be initialized as np.inf
            #such that error after the first epoch will be forced to be lower than it
            if self.early_stopping and self.error_val[-1]>=self.error_val[-2]:
                self.w=self.w_epoch[-2]
                break
    def update_weight(self, i):
        self.w=self.w+self.learning_rate*self.x_train[i]*self.y_train[i]
    def predict_one(self, x):
        prediction=sign(np.sum(np.multiply(x, self.w)))
        return prediction
    def predict_more(self, x_s):
        predictions=np.apply_along_axis(self.predict_one, axis=1, arr=x_s)
        return predictions
    def test_sample(self, x_sample, y_sample):
        if self.bias and x_sample.shape[1]!=self.n_features:
            x_sample = np.column_stack([x_sample, np.ones(x_sample.shape[0])])
        error=np.sum(np.ones(x_sample[0])-np.multiply(self.predict_more(x_sample), y_sample))
        return error