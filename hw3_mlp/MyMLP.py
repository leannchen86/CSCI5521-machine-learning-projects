import numpy as np


# normalize raw data by (x-mean)/std
def normalize(x, mean=None, std=None):
    if mean is None:
        mean = np.mean(x, axis=0).reshape(1,-1)
        std = np.std(x, axis=0).reshape(1,-1)
    x = (x-mean)/(std+1e-5)
    return x, mean, std


# ------------------- Please Implment this function ------------------------------
# Input: a list of labels, n*1
# Output: one hot encoding of the labels
# For example, if there are 3 class, and the labels for the data are [0,0,1,1,0,2],
# the one hot encoding should be [[1,0,0],[1,0,0],[0,1,0],[0,1,0],[1,0,0],[0,0,1]]
def process_label(label):
    # placeholders
    one_hot = np.zeros([len(label),10])
    one_hot = np.eye(10)[label]

    return one_hot


# ------------------- Please Implment this function ------------------------------
# input: intermediate features (n,d) 
# output: resuts of pluging the value into (e^-x-e^x)/(e^-x+e^x), you can use np.exp()
# the output should have the same shape of the input (n,d)
def tanh(x):
    # implement the hyperbolic tangent activation function for hidden layer
    out = np.zeros_like(x)
    # preprocess x to boost the performance, no need to modify
    x = np.clip(x,a_min=-100,a_max=100)
    out = -(np.exp(-x) - np.exp(x))/ (np.exp(-x) + np.exp(x))

    return out 


# ------------------- Please Implment this function ------------------------------
# input: intermediate features (n,d) 
# output: resuts of pluging the value into (e^xi)/(sum_i e^xi), you can use np.exp()
# the output should have the same shape of the input (n,d)
# for example, if input is [[1,2],[1,3]], output should be 
# [[e^1/(e^1+e^2), e^2/(e^1+e^2)], [e^1/(e^1+e^3), e^1/(e^1+e^3)]]
def softmax(x):
    # implement the softmax activation function for output layer
    out = np.zeros_like(x)
    out = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    return out


class MLP:
    def __init__(self,num_hid):
        # initialize the weights
        self.num_hid = num_hid
        self.lr = 5e-3 # 5e-3
        self.w = np.random.random([64,num_hid])
        self.w0 = np.random.random([1,num_hid])
        self.v= np.random.random([num_hid,10])
        self.v0 = np.random.random([1,10])

    # This function centers around the training process
    def fit(self,train_x,train_y, valid_x, valid_y):
        # counter for recording the number of epochs without improvement
        count = 0
        best_valid_acc = 0

        """
        Stop the training if there is no improvment over the best validation accuracy for more than 100 iterations
        """
        while count<=50:
            # training with all samples (full-batch gradient descents)
            # implement the forward pass for all samples
            z, y = self.forward(train_x)

            # implement the backward pass (backpropagation)
            # compute the gradients w.r.t. different parameters
            gra_v = self.dEdv(z, y, train_y)
            gra_v0 = self.dEdv0(y, train_y)
            gra_w = self.dEdw(z, y, train_x, train_y)
            gra_w0 = self.dEdw0(z, y, train_y)

            # update the parameters
            self.update(gra_w, gra_w0, gra_v, gra_v0)

            # evaluate on validation data
            predictions = self.predict(valid_x)
            valid_acc = np.count_nonzero(predictions.reshape(-1)==valid_y.reshape(-1))/len(valid_x)

            # compare the current validation accuracy with the best one
            if valid_acc>best_valid_acc:
                best_valid_acc = valid_acc
                count = 0
            else:
                count += 1

        return best_valid_acc

    # ------------------- Please Implment this function ------------------------------
    # the forward pass map the input x to output y with 2 layers
    # input: input features x (n,64) 
    # output: z intermediate output (n, num_hid) 
    #         y final output        (n,10)
    # z = tanh(xw+w0) x:(n,64), w: (64, num_hid), w0: (1, num_hid), z: (n, num_hid) 
    # y = softmax(zv+v0) v:(num_hid, 10), v0: (1, 10), y:(n,10)
    def forward(self, x):
        # placeholders
        z = np.zeros([len(x), self.num_hid])
        y = np.zeros([len(x), 10])
        z = tanh(x@self.w + self.w0)
        y = softmax(z@self.v + self.v0)

        return z, y

    # ------------------- Please Implment all the following function -------------------
    # In the following, you are going to implement the most important part - compute update
    # rules. As we have already learn that in the lecture, the update rule is basically the 
    # multiplicate between multiple partial derivatives.
    # Assume 
    # r is the labels (e.g train y)
    # t = xw+w0 (n, num_hid)
    # z = tanh(t) (n, num_hid)
    # o = zv+v0 (n,10)
    # y = softmax(o) (n, 10)
    # We can have the derivative for each parameters
    # gra_v = dE/dy * dy/do * do/dv  (1, num_hid)
    # gra_v0 = dE/dy * dy/do * do/dv0  (1, 10)
    # gra_w = dE/dy * dy/do * do/dw  (1, 64)
    # gra_w0 = dE/dy * dy/do * do/dw0  (1, num_hid)

    # Next we demonstrate a concrete example to help you start with the derivatives to
    # get these formula
    # ------------- gra_v -------------------------------------------------------------
    # dE/dv = [[dE/dv11, dE/dv12, ..., dE/dv1k], 
    #          [dE/dv21, dE/dv22, ..., dE/dv2k],
    #           ....
    #          [dE/dv101, dE/dv102, ..., dE/dv10k]] (10, num_hid) vector
    # To get each term, we select row j (dE/dvj) to appraoch, dE/dvj = dE/dy * dy/vj (where vj is (1, num_hid))
    #
    # dE/dy = [dE/dy1, dE/dy2, .... , dE/dy10] (1, 10)
    # dy/dv = [[dy1/dv1, dy1/dv2, ..., dy1/dv10], 
    #          [dy2/dv1, dy2/dv2, ..., dy2/dv10],
    #           ....
    #          [dy10/dv1, dy10/dv2, ..., dy10/dv10]]  v_j denotes a (1, num_hid) vector
    # 
    # each term in dE/dv will be: dE/dvj = sum_i dE/dyi * dyi/dvj 
    # for any term in the dy/dv, 
    # e.g, dyi/dvj = sum_k dyi/dok * dok/dvj = sum_k yi(delta_ik - yk)*(dok/dvj)
    # since only when k=j dok/dvj != 0, the term is simplified to yi(delta_ij-yj)*z
    # pluging dyi/dvj into  dE/dvj = sum_i dE/dyi * dyi/dvj = -sum_i ri/yi * yi(1-yj)*z
    # - sum_i ri/yi * yi(delta_ij-yj)*z = - (sum_i ri*delta_ij*z - sum_i ri*yj*z)
    # since sum_i yi = 1 (one hot encoding sum up to 1), delta_ij = 0 if i != j
    # the term is simplified to dE/dvj = -(rj*zj - yj*z) = (yj-rj)*z
    # as (ri-yi) has shape (1, n), z has shape (n, num_hid), the term should be z.T(yj-rj)
    # plugging this term back to dE/dv
    # we have dE/dv = [[z.T(y1-r1)], 
    #                  [z.T(y2-r2)],
    #                   ...
    #                  [z.T(y10-r10)]], where zi has shape (n, num_hid),
    # finally, dE/dv can be formuated as 
    # dE/dv = z.T@(y-r)

    # Input: z, output of the intermediate layer (n, num_hid) 
    #        y, output of the last layer (n, 10)
    #        r, gt one-hot labels (n, 10)
    # Output: gra_v, (num_hid, 10)
    def dEdv(self, z, y, r):
        # placeholder
        out = np.zeros_like(self.v)
        out = z.T@(y-r)

        return out

    # the only difference between v and v0 is that you need to replace z with
    # a (n, 1) vector, whose entries are 1
    # Input: z, output of the intermediate layer (n, num_hid) 
    #        y, output of the last layer (n, 10)
    #        r, gt one-hot labels (n, 10)
    # Output: out, gra_v0, (1, 10)
    # c = np.ones(n,1)
    # gra_v0 = c.T@(y-r) or (y-r).sum(axis=0)
    def dEdv0(self, y, r):
        # placeholder
        out = np.zeros_like(self.v0)
        out = np.sum(y-r, axis=0)

        return out

    # The following two derivatives are left for you to derive by yourself, by adding extra terms into
    # the derivative
    # hint: take care of the operations between different vectors/matrices, determine whether it should 
    # be elementwise(*) or matrix multication(@)
    # Input: z, output of the intermediate layer (n, num_hid) 
    #        y, output of the last layer (n, 10) 
    #        x, input of first layer (n, 64)
    #        r, gt one-hot labels (n, 10)
    # Output: out, gra_w, (64, num_hid)
    def dEdw(self, z, y, x, r):
        # placeholder
        out = np.zeros_like(self.w)
        # w_inner = ((y-r)@(z.T@(y-r)).T)*(1-z**2)
        w_inner = ((y-r)@(self.v.T))*(1-z**2)
        out = (w_inner.T @ x).T

        return out

    # Input: z, output of the intermediate layer (n, num_hid) 
    #        y, output of the last layer (n, 10) 
    #        r, gt one-hot labels (n, 10)
    # Output: out, gra_w, (1, num_hid)
    def dEdw0(self, z, y, r):
        # placeholder
        out = np.zeros_like(self.w0)
        # w_inner = ((y-r)@(z.T@(y-r)).T)*(1-z**2)
        w_inner = ((y-r)@(self.v.T))*(1-z**2)
        out = np.sum(w_inner, axis=0)

        return out

    # Input: gra_w,  
    #        gra_w0,  
    #        gra_v, 
    #        gra_v0, four gradients
    # Output: no return, direcly update the class parameters self.w, self.w0, .....
    # e.g self.w = self.w - self.lr*gra_w
    def update(self, gra_w, gra_w0, gra_v, gra_v0):
        self.w -= self.lr*gra_w
        self.w0 -= self.lr*gra_w0
        self.v -= self.lr*gra_v
        self.v0 -= self.lr*gra_v0

        return self.w, self.w0, self.v, self.v0

    # ----------------------------- end of your implemented functions -------------------------

    def predict(self,x):
        # generate the predicted probability of different classes
        z = tanh(x.dot(self.w) + self.w0)
        y = softmax(z.dot(self.v) + self.v0)
        # convert class probability to predicted labels
        y = np.argmax(y,axis=1)

        return y

    def get_hidden(self,x):
        # extract the intermediate features computed at the hidden layers
        z = tanh(x.dot(self.w) + self.w0)
        return z
