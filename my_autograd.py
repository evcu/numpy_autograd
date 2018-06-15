import numpy as np

class Variable():
    __counter = 0
    def __init__(self,data,is_leaf=True,backward_fun=None):
        if backward_fun is None and not is_leaf:
            raise ValueError('non leaf nodes require backward_fun')
        if np.isscalar(data):
            data = np.ones(1)*data
        if not isinstance(data,np.ndarray):
            raise ValueError(f'data should be of type "numpy.ndarray" or a scalar,but received {type(data)}')
        self.data = data
        self.id = Variable.__counter
        Variable.__counter += 1
        self.is_leaf = is_leaf
        self.prev = []
        self.backward_fun = backward_fun
        self.zero_grad()

    def backward(self):
        self.backward_fun(dy=self.grad)

    def zero_grad(self):
        self.grad = np.zeros(self.data.shape)

    def step(self,lr):
        self.data -= lr*self.grad


    def __repr__(self):
        return f'Variable(id:{self.id},prev:{list(map(lambda a:a.id,self.prev))},is_leaf:{self.is_leaf})\n'



def plus(a,b):
    if not (isinstance(a,Variable) and isinstance(b,Variable)):
        raise ValueError('a,b needs to be a Variable instance')
    def b_fun(dy):
        b.grad += dy
        a.grad += dy

    res = Variable(a.data+b.data,is_leaf=False,backward_fun=b_fun)
    res.prev.extend([a,b])
    return res

def plus_bcast(a,b):
    """
    a being a matrix(mini-batch output m*n)
    b being a vector(bias n)
    """
    if not (isinstance(a,Variable) and isinstance(b,Variable)):
        raise ValueError('a,b needs to be a Variable instance')
    def b_fun(dy):
        b.grad += dy.sum(axis=0)
        a.grad += dy

    res = Variable(a.data+b.data,is_leaf=False,backward_fun=b_fun)
    res.prev.extend([a,b])
    return res


# def absolute(a):
#     if not (isinstance(a,Variable)):
#         raise ValueError('a needs to be a Variable')
#     def b_fun(dy=1):
#         mask = np.ones(dy.shape)
#         mask[a.data<0]=-1
#         a.grad += mask*dy
#
#     res = Variable(np.abs(a.data),is_leaf=False,backward_fun=b_fun)
#     res.prev.append(a)
#     return res

def minus(a,b):
    if not (isinstance(a,Variable) and isinstance(b,Variable)):
        raise ValueError('a,b needs to be a Variable instance')
    def b_fun(dy):
        b.grad += -dy
        a.grad += dy
    res = Variable(a.data-b.data,is_leaf=False,backward_fun=b_fun)
    res.prev.extend([a,b])
    return res

def sumel(a):
    if not (isinstance(a,Variable)):
        raise ValueError('a needs to be a Variable')
    def b_fun(dy=1):
        a.grad += np.ones(a.data.shape)*dy

    res = Variable(np.sum(a.data),is_leaf=False,backward_fun=b_fun)
    res.prev.append(a)
    return res

def transpose(a):
    if not (isinstance(a,Variable)):
        raise ValueError('a needs to be a Variable')
    def b_fun(dy=1):
        a.grad += dy.T

    res = Variable(a.data.T,is_leaf=False,backward_fun=b_fun)
    res.prev.append(a)
    return res

def dot(a,b):
    if not (isinstance(a,Variable) and isinstance(b,Variable)):
            raise ValueError('a,b needs to be a Variable instance')
    def b_fun(dy):
        if np.isscalar(dy):
            dy = np.ones(1)*dy
        a.grad += np.dot(dy,b.data.T)
        b.grad += np.dot(a.data.T,dy)
    res = Variable(np.dot(a.data,b.data),is_leaf=False,backward_fun=b_fun)
    res.prev.extend([a,b])
    return res

def multiply(a,b):
    if not (isinstance(a,Variable) and isinstance(b,Variable)):
            raise ValueError('a,b needs to be a Variable instance')
    def b_fun(dy):
        if np.isscalar(dy):
            dy = np.ones(1)*dy
        a.grad += np.multiply(dy,b.data)
        b.grad += np.multiply(dy,a.data)
    res = Variable(np.multiply(a.data,b.data),is_leaf=False,backward_fun=b_fun)
    res.prev.extend([a,b])
    return res

def c_mul(a,c):
    if not (isinstance(a,Variable) and isinstance(c,(int, float))):
        raise ValueError('a needs to be a Variable, c needs to be one of (int, float)')
    def b_fun(dy=1):
        a.grad += dy*c
    res = Variable(a.data*c,is_leaf=False,backward_fun=b_fun)
    res.prev.append(a)
    return res

def relu(a):
    if not (isinstance(a,Variable)):
        raise ValueError('a needs to be a Variable')
    def b_fun(dy=1):
        a.grad[a.data>0] += dy[a.data>0]

    res = Variable(np.maximum(a.data, 0),is_leaf=False,backward_fun=b_fun)
    res.prev.append(a)
    return res

def __top_sort(var):
    vars_seen = set()
    top_sort = []
    def top_sort_helper(vr):
        if (vr in vars_seen) or vr.is_leaf:
            pass
        else:
            vars_seen.add(vr)
            for pvar in vr.prev:
                top_sort_helper(pvar)
            top_sort.append(vr)
    top_sort_helper(var)
    return top_sort

def backward_graph(var):
    if not isinstance(var,Variable):
        raise ValueError('var needs to be a Variable instance')
    tsorted = __top_sort(var)

    var.grad=np.ones(var.data.shape)
    for var in reversed(tsorted):
        var.backward()


class LinearLayer():
    def __init__(self,features_inp,features_out):
        super(LinearLayer, self).__init__()
        std = 1.0/features_inp
        self.w = Variable(np.random.uniform(-std,std,(features_inp,features_out)))
        self.b = Variable(np.random.uniform(-std,std,features_out))

    def forward(self, inp):
        return plus_bcast(dot(inp,self.w),self.b)

    def zero_grad(self):
        self.w.zero_grad()
        self.b.zero_grad()

    def step(self,lr):
        self.w.step(lr)
        self.b.step(lr)
