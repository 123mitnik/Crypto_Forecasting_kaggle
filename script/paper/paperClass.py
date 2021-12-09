import numpy as np

class paperCI:
    '''
    generate mle, and test statistics given a theta_0(thetaci) for panel ar(1)
    '''
    def __init__(self, df):
        # initial attributes
        self.Y = df #NXT panel array
        self.N, self.T= df.shape

    def __call__(self, thetaci, showmle = False , *args, **kwargs):
        self.thetaci = thetaci
        self.Y0 = np.concatenate((np.zeros(self.N).reshape(self.N, 1), self.Y[:, 0:(self.T-1)]), axis=1)

        # mle by double summation, row means:axis=1
        up = sum(sum((self.Y0 - self.Y0.mean(axis=1).reshape(self.N, 1))*(self.Y - self.Y.mean(axis=1).reshape(self.N, 1))))
        down = sum(sum((self.Y0 - self.Y0.mean(axis=1).reshape(self.N, 1))**2))
        thetahat = up/down
        sigma2hat = np.mean(((self.Y - self.Y.mean(axis=1).reshape(self.N,1))-
                          thetahat*(self.Y0 - self.Y0.mean(axis=1).reshape(self.N,1)))**2)
        
        self.mle = (thetahat, sigma2hat)
        if showmle:
        	print(f'MLE ={self.mle[0]}')

        # normalized hahn and kuersteiner statistics
        stable = np.sqrt(self.N*self.T)*(self.mle[0]-self.thetaci + ((1+self.mle[0])/self.T))
        self.HKstable = stable/np.sqrt(1-self.thetaci**2)


        nonstable = (np.sqrt(self.N)*self.T)*(self.mle[0]-self.thetaci + (3/(self.T+1)))
        self.HKnonstable = nonstable/np.sqrt(51/5)

        # self normalized t stat, with B and V
        A = np.zeros((self.T,self.T))
        dd = np.array([self.thetaci**x for x in np.arange(self.T-2,-1,-1)])
        for r in range(self.T):
            if r > 0:
                A[r, np.arange(r)] = dd[(self.T -r-1):(self.T-1)]
        one = np.array([1]*self.T).reshape(self.T,1)#column ones vector
        H = np.diag([1]*self.T) - (1/self.T)*one@one.T
        D_mat = H@A
        G_mat = D_mat + (3/(self.T+1))*D_mat.T@D_mat

        down = sum(sum((self.Y0 - self.Y0.mean(axis=1).reshape(self.N, 1))**2))**2
        up = self.N*self.mle[1]*sum(np.diag(G_mat))
        self.B = (-3/(self.T+1), -3/(self.T+1) + up/down)#hk bias, self-bias

        up = self.mle[1]**2 *self.N*self.T**2 * 17/60
        M = (G_mat + G_mat.T)/2
        upci = 2*self.N*self.mle[1]**2 *sum(np.diag(M@M))
        self.V = (up/down, upci/down)#hk variance, self-variance

        ss = (self.mle[0]-self.thetaci - self.B[0])/np.sqrt(self.V[0])#hk-normalization
        ssci = (self.mle[0]-self.thetaci - self.B[1])/np.sqrt(self.V[1])#self-normalization
        self.tstat = (ss, ssci)# we use the second one

        #  M statistics
        T1 = self.T-1
        y = self.Y[:, -T1:]
        y0 = self.Y[:, range(T1)]
        thetarho = self.mle[0]
        sigma2hat = np.mean((y-y.mean(axis=1).reshape(self.N, 1)-
                                              thetarho*(y0-y0.mean(axis=1).reshape(self.N, 1)))**2)
        self.Msigma2 = sigma2hat

        y = self.Y[:, 1:(self.T-2)]
        y0 = self.Y[:, range(self.T-3)]
        gammahat = sigma2hat/(self.N*self.T)* (sum(sum((y0-y)**2)) + sum(y[:, -1]**2))

        deltaY = (self.Y - self.Y0)[:, -(self.T-2):]
        deltaY0 = (self.Y - self.Y0)[:, -(self.T-1):-1]
        self.M = (1/(gammahat*np.sqrt(self.N*self.T)))*sum(sum(self.Y[:,range(self.T-2)] * (deltaY-self.thetaci*deltaY0)))
        
        return self.tstat[1],self.HKstable,self.HKnonstable,self.M











