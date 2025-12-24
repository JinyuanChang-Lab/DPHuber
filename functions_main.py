import numpy as np
import numpy.random as rgt


#########################################################
#---------------------- basic functions  ----------------
######################################################### 
# Compute the Median Absolute Deviation (MAD), a robust measure of scale.
def mad(x):
    return np.median(abs(x - np.median(x)))*1.4826

# Huber score function  
def huber_score(x, c): 
    return np.where(abs(x)<=c, x, c*np.sign(x))
    
# Clip covariates based on their l2 norms
def clipping_l2(X, B):
    '''
        Covariates Clipping: Scales each row of X to have a maximum l2 norm of B.
    '''
    n = X.shape[0]
    x_norm = np.array([X[i,:].dot(X[i,:])**0.5 for i in range(n)])
    trun_weight = np.minimum(1, B/x_norm)
    return (X.T*(trun_weight)).T

# Clip covariates based on the maximum absolute value of each row.
def clipping_inf(X, B):
    '''
        Covariates Clipping: based on infinity norm.
    '''
    n = X.shape[0]
    x_max_norm = np.array([np.max(np.abs(X[i,:])) for i in range(n)])
    trun_weight = np.minimum(1, B/x_max_norm)
    return (X.T*(trun_weight)).T
 
# Hard Thresholding (HT) to keep only the largest s elements.
def ht(v, s):
    d = len(v) 
    top_indices = np.argsort(-np.abs(v))[:s]    
    v_S = np.zeros(d)
    v_S[top_indices] = v[top_indices]
    return v_S

# Noisy  Hard Thresholding (Noisy HT) with added Laplace noise for differential privacy. 
def noisyht( v, s, epsilon, delta, lambda_scale, rng=None):
    if rng is None:
        rng = rgt
    d = len(v)
    if s > d:
        raise ValueError("s cannot be larger than dimension.")

    S  = []  
    sigma = lambda_scale * 2 * np.sqrt(5 * s * np.log(1/delta)) / epsilon
    for _ in range(s):
        w= rng.laplace(0.0, sigma, size=d)
        candidates = [(abs(v[j])+w[j]  , j) for j in range(d) if j not in S]
        _, j_max = max(candidates, key=lambda x: x[0])
        S.append(j_max)
   
    v_S = np.zeros(d)
    noise = rng.laplace(0.0, sigma, size=d)
    for j in S:
        v_S[j] = v[j] + noise[j]
    return v_S 
  
#########################################################
#-------- main: Huber regression model class  -----------
#########################################################
class Huber():
    def __init__(self, X, Y, intercept=True):
        '''
        Arguments
        ---------
            X : n by p numpy array of covariates; each row is an observation vector.
            
            Y : n by 1 numpy array of response variables. 
            
            intercept : logical flag for adding an intercept to the model.
        '''
        n = len(Y)
        self.Y = Y
        self.mX, self.sdX = np.mean(X, axis=0), np.std(X, axis=0)
        self.itcp = intercept
        if intercept:
            self.X = np.concatenate([np.ones((n,1)), X], axis=1)
            self.X1 = np.concatenate([np.ones((n,1)), (X - self.mX)/self.sdX], axis=1)
        else: 
            self.X, self.X1 = X, (X - self.mX)/self.sdX
    
    #----------------------------------------
    #    Low-dimensional  Huber regression 
    #----------------------------------------
    # non-private
    def gd(self, 
           tau = 5, 
           lr=1, 
           beta0=np.array([]),
           T = 1,
           standardize=False, adjust=False):
        '''
        Perform gradient descent to minimize Huber loss in low dimensions.

        Parameters:
        tau: float, robust parameter used in Huber loss. 
        lr : float, learning rate for gradient updates.
        beta0 : numpy array, initial coefficients.    
        T : int, maximum number of iterations.
        standardize : bool, whether to standardize covariates.
        adjust : bool, whether to adjust coefficients back to original scale.
        '''
        
        if len(beta0) == 0:
            beta0 = np.zeros(self.X.shape[1]) 
       
        if standardize: X = self.X1
        else: X = self.X
    
        beta_seq = np.zeros([X.shape[1], int(T)+1])
        beta_hat = beta0.copy()
        res = self.Y - X @ beta_hat 
        beta_seq[:,0] = beta_hat
        count = 0 
        while count < int(T): 
            beta_hat += lr * X.T @ huber_score(res, tau) / X.shape[0]
            beta_seq[:, count+1] = beta_hat
            res = self.Y - X @ beta_hat 
            count += 1

        if standardize and adjust:
            beta_hat[self.itcp:] = beta_hat[self.itcp:]/self.sdX
            beta_seq[self.itcp:,] = beta_seq[self.itcp:,]/self.sdX[:,None]
            if self.itcp: 
                beta_hat[0] -= self.mX.dot(beta_hat[1:])
                beta_seq[0,:] -= self.mX.dot(beta_seq[1:,])

        return {'beta': beta_hat,
                'beta_seq': beta_seq,
                'residuals': res,
                'robust': tau,
                'niter': count}
    
    # private
    def noisygd(self, 
                tau=5,
                sigma_scale=1,
                B=1, 
                T=50,
                lr=1, 
                beta0=np.array([]), 
                standardize=False, adjust=False):
        '''
        Implements Noisy Gradient Descent (NoisyGD) with Huber loss for robust regression.

        Parameters:
        ----------
        tau : float, robust parameter used in Huber loss. 
        sigma_scale : float, standard deviation scale of injected Gaussian noise. 
        B : float, Clipping bound for the covariates.
        T : int, number of gradient descent iterations.
        lr : float, learning rate for gradient descent.
        beta0 : ndarray, initial coefficients for the regression model. Defaults to zeros. 
        standardize : bool, whether to standardize covariates before training.
        adjust : bool, whether to adjust coefficients back to the original scale.
        ''' 

        if len(beta0) == 0:
            beta0 = np.zeros(self.X.shape[1]) 

        if standardize: X = self.X1
        else: X = self.X
        trun_X = clipping_l2(X, B)

        beta_seq = np.zeros([X.shape[1], int(T)+1])
        beta_hat = beta0.copy()
        res = self.Y - X @ beta_hat
        beta_seq[:,0] = beta_hat
        count = 0
        while count < int(T): 
            sigma = tau * sigma_scale
            n = X.shape[0] 
            diff = (lr/n) * (trun_X.T @ huber_score(res, tau) \
                             + sigma * rgt.standard_normal(X.shape[1]))
            beta_hat += diff
            beta_seq[:, count+1] = beta_hat 
            res = self.Y - X @ beta_hat
            count += 1

        if standardize and adjust:
            beta_hat[self.itcp:] = beta_hat[self.itcp:]/self.sdX
            beta_seq[self.itcp:,] = beta_seq[self.itcp:,]/self.sdX[:,None]
            if self.itcp: 
                beta_hat[0] -= self.mX.dot(beta_hat[1:])
                beta_seq[0,:] -= self.mX.dot(beta_seq[1:,])

        return {'beta': beta_hat,
                'beta_seq': beta_seq,
                'residuals': res,
                'robust': tau,
                'niter': count}
    
    #----------------------------------------
    #    High-dimensional  Huber regression 
    #----------------------------------------
    # non-private
    def gd_highdim(self, 
                    s, 
                    lr, 
                    T,  
                    tau=None, 
                    beta0=np.array([]),
                    standardize=False, adjust=False):
        '''
        Performs high-dimensional gradient descent using adaptive Huber loss.

        Parameters:
        ----------
        s : int, sparsity level of the solution (number of non-zero coefficients).
        lr : float, learning rate for gradient descent.
        T : int, maximum number of iterations.
        tau : float, robust parameter for Huber loss.
        beta0 : ndarray, initial coefficients (default: zeros).
        standardize : bool, whether to standardize covariates.
        adjust : bool, whether to adjust coefficients to the original scale.
        '''
        if len(beta0) == 0:
            beta0 = np.zeros(self.X.shape[1]) 

        if standardize: X = self.X1
        else: X= self.X
        n = X.shape[0]
   
        beta_seq = np.zeros([X.shape[1], T+1])
        beta_hat = beta0.copy()
        beta_seq[:,0] = beta_hat
        res = self.Y - X @ beta_hat
        count = 0     
        while count < T:  
            beta_hat += (lr/n) * (X.T @ huber_score(res, tau)) 
            beta_hat  =  ht(beta_hat, s=s ) 
            res = self.Y-X @ beta_hat
            beta_seq[:, count+1] = beta_hat
            count += 1

        if standardize and adjust:
            beta_hat[self.itcp:] = beta_hat[self.itcp:]/self.sdX
            beta_seq[self.itcp:,] = beta_seq[self.itcp:,]/self.sdX[:,None]
            if self.itcp: 
                beta_hat[0] -= self.mX.dot(beta_hat[1:])
                beta_seq[0,:] -= self.mX.dot(beta_seq[1:,])

        return  {'beta': beta_hat,
                'beta_seq': beta_seq,
                'residuals': res, 
                'niter': count}
    

    def noisygd_highdim(self, 
                            s, 
                            lr, 
                            T,                          
                            tau=5,
                            beta0=np.array([]), 
                            B_high=1, 
                            epsilon_scale =1, 
                            delta_scale =1, 
                            standardize=False, adjust=False):
        '''
        Perform noisy gradient descent for high-dimensional sparse regression.
        
        Parameters:
            s : int, sparsity level for the regression coefficients.
            lr : float, learning rate for gradient updates.
            T : int, number of iterations.
            epsilon_scale, delta_scale : float, privacy parameters controlling noise level at each iteration (see Algorithm 3 for details). 
            tau : float, robust parameter for huber loss function 
            beta0 : ndarray, initial guess for the regression coefficients.
            B_high : float, clipping bound. 
            standardize : bool, whether to standardize the design matrix.
            adjust : bool, whether to adjust for mean and standard deviation in output. 
        '''
        if len(beta0) == 0:
            beta0 = np.zeros(self.X.shape[1]) 
            
        if standardize: X = self.X1
        else: X= self.X
        n = X.shape[0]
        if self.itcp:
            trun_X1 = np.concatenate([np.ones((n,1)),clipping_inf(X[:,1:] , B_high)], axis=1)
        
        beta_seq = np.zeros([X.shape[1], T+1])
        beta_hat = beta0.copy()
        beta_seq[:,0] = beta_hat
        # Compute initial residuals
        res = self.Y - X @ beta_hat
        count = 0 
        while count < T: 
            beta_hat += (lr/n) * (trun_X1.T @ huber_score(res, tau))
            beta_hat =  noisyht(beta_hat, s=s,epsilon=epsilon_scale, delta=delta_scale, lambda_scale=2*(lr/n)*B_high*tau,rng=rgt ) 
            res = self.Y-X @ beta_hat 
            beta_seq[:, count+1] = beta_hat
            count += 1

        if standardize and adjust:
            beta_hat[self.itcp:] = beta_hat[self.itcp:]/self.sdX
            beta_seq[self.itcp:,] = beta_seq[self.itcp:,]/self.sdX[:,None] 
            if self.itcp: 
                beta_hat[0] -= self.mX.dot(beta_hat[1:])
                beta_seq[0,:] -= self.mX.dot(beta_seq[1:,])        

        return  {'beta': beta_hat, 
                'beta_seq': beta_seq, 
                'residuals':  res, 
                'niter': count}
 