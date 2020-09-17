import numpy as np
import operator

class Tensor():  
    def __init__(self,shape):
        self.shape=shape
        self.totalSize = reduce(operator.mul,shape,1) 
        self.storage = np.ones(self.totalSize)

    # assume b is smaller
    @staticmethod
    def isSuffix(a,b):
        for i in range(0, len(b), 1):
            if not (b[len(b)-i-1]==a[len(a)-i-1]):
                print(b[len(b)-i-1],a[len(a)-i-1]) 
                return False
        return True
        
    # b should be smaller
    def _multiply(self,b):
        print("shapes", self.shape, b.shape)
        loneDimsLen = len(self.shape)-len(b.shape)           
        print("lone dims", loneDimsLen)

        bTotalSize = reduce(operator.mul, b.shape,1)
        aTotalSize = reduce(operator.mul, self.shape,1)
        print("total sizes", aTotalSize, bTotalSize)
        multiplier = bTotalSize
        c = Tensor(a.shape)
        for i in range(0, aTotalSize, bTotalSize):
            for j in range(bTotalSize):
                index = i+j
                c.storage[index]=self.storage[index]+b.storage[j]
        print(c.storage)
                
            
    def __mul__(self, o):
        print(o.__class__.__name__)
        if o.__class__.__name__=='Tensor':
            if len(self.shape)>=len(b.shape):
                isSuffixOfLeft = self.isSuffix(self.shape,b.shape)
                self._multiply(o)
            else:
                isRightSuffix = self.isSuffix(b,a)
       

a = Tensor([2,3,1,2])
b = Tensor([3,1,2])
print(a.totalSize)
print(a.storage)
print(Tensor.isSuffix([2,3], [2,3,1]))
print(Tensor.isSuffix([2,3,1], [3,1]))
a*b
