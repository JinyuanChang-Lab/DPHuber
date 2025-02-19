import numpy as np
import numpy.random as rgt

### basic functions 
# Compute the Median Absolute Deviation (MAD), a robust measure of scale.
def mad(x):
    return np.median(abs(x - np.median(x)))*1.4826

# Huber score function  
def huber_score(x, c): 
    return np.where(abs(x)<=c, x, c*np.sign(x))
    
# Clip covariates based on their l2 norms
def clipping(X, B):
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
def noisyht( v, s, mu, delta, lambda_scale):
    d = len(v)
    S  = set()  
    for _ in range(s):
        w= np.random.laplace(0, lambda_scale*2*np.sqrt( 5*s*np.log(1/delta))/mu, (d,))
        candidates = [(abs(v[j])+w[j]  , j) for j in range(d) if j not in S]
        _, j_max = max(candidates, key=lambda x: x[0])
        S.add(j_max)
   
    v_S = np.zeros(d)
    noise = np.random.laplace(0, lambda_scale*2*np.sqrt(5* s*np.log(1/delta))/mu, (d,))
    for j in S:
        v_S[j] = v[j] + noise[j]
    return v_S
 
# Project a vector onto a ball of radius R.
def projection(v, R):
    norm_v = np.linalg.norm(v, 2)   
    if norm_v <= R:
        return v
    else:
        return (R / norm_v) * v  

### main: Huber regression model class.
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
    
    # Low-dimensional adaptive Huber regression.
    def gd(self, robust=5, robust_scale=True, 
           lr=1, beta0=np.array([]), res=np.array([]),
           standardize=True, adjust=True, max_niter=1e3):
        '''
        Perform gradient descent to minimize Huber loss in low dimensions.

        Parameters:
        robust : float, adaptive parameter for Huber loss.
        robust_scale : bool, whether to use MAD as the standard deviation estimation method
        lr : float, learning rate for gradient updates.
        beta0 : numpy array, initial coefficients.
        standardize : bool, whether to standardize covariates.
        adjust : bool, whether to adjust coefficients back to original scale.
        max_niter : int, maximum number of iterations.
        '''
        if len(beta0) == 0:
            beta0 = np.zeros(self.X.shape[1])
            if self.itcp: beta0[0] = np.mean(self.Y)
       
        if standardize: X = self.X1
        else: X = self.X
        res = self.Y - X.dot(beta0)
        if robust_scale:
            tau = mad(res)*robust
        else:
            tau = np.std(res)*robust

        beta_seq = np.zeros([X.shape[1], int(max_niter)+1])
        beta_seq[:,0] = beta0
        count = 0
        while count < int(max_niter):
            beta0 += lr * X.T @ huber_score(res, tau) / X.shape[0]
            beta_seq[:, count+1] = np.copy(beta0)
            res = self.Y - X @ beta0
            if robust_scale:
                tau = mad(res)*robust
            else:
                tau = np.std(res)*robust
            count += 1

        if standardize and adjust:
            beta0[self.itcp:] = beta0[self.itcp:]/self.sdX
            beta_seq[self.itcp:,] = beta_seq[self.itcp:,]/self.sdX[:,None]
            if self.itcp: 
                beta0[0] -= self.mX.dot(beta0[1:])
                beta_seq[0,:] -= self.mX.dot(beta_seq[1:,])

        return {'beta': beta0,
                'beta_seq': beta_seq[:,:count+1],
                'residuals': res,
                'robust': tau,
                'niter': count}
    

    def noisygd(self, robust=5,
                    GDP=True, mu=0.5, delta=1e-2, B=1, T=50,
                    lr=1, beta0=np.array([]), res=np.array([]),
                    standardize=True, adjust=True):
        '''
        Implements Noisy Gradient Descent (NoisyGD) with Huber loss for robust regression.

        Parameters:
        ----------
        robust : float, used in Huber loss.
        GDP : bool, if True, applies Gaussian Differential Privacy (GDP) mechanism.
        mu : float, privacy parameter(default: 0.5).
        delta : float, privacy parameter delta (default: 1e-2).
        B : float, optional
            Clipping bound for the covariates (default: 1).
        T : int, number of gradient descent iterations (default: 50).
        lr : float, learning rate for gradient descent (default: 1).
        beta0 : ndarray, initial coefficients for the regression model. Defaults to zeros.
        res : ndarray, residuals, initialized as empty by default.
        standardize : bool, whether to standardize covariates before training.
        adjust : bool, whether to adjust coefficients back to the original scale.
        '''
        if len(beta0) == 0:
            beta0 = np.zeros(self.X.shape[1])
            if self.itcp: beta0[0] = np.mean(self.Y)

        if standardize: X = self.X1
        else: X = self.X
        res = self.Y - X.dot(beta0) 
        tau = robust
        n = X.shape[0] 

        if GDP:
            sigma = 2*tau*B*np.sqrt(T)/mu 
        else:
            sigma = 2*tau*B*(T*( (2*np.log(1.25*T/delta))**0.5 ))/mu
        trun_X = clipping(X, B)

        beta_seq = np.zeros([X.shape[1], int(T)+1])
        beta_seq[:,0] = beta0
        count = 0
        while count < int(T):
            diff = (lr/n) * (trun_X.T @ huber_score(res, tau) \
                             + sigma * rgt.standard_normal(X.shape[1]))
            beta0 += diff
            beta_seq[:, count+1] = np.copy(beta0)
            res = self.Y - X @ beta0
            count += 1

        if standardize and adjust:
            beta0[self.itcp:] = beta0[self.itcp:]/self.sdX
            beta_seq[self.itcp:,] = beta_seq[self.itcp:,]/self.sdX[:,None]
            if self.itcp: 
                beta0[0] -= self.mX.dot(beta0[1:])
                beta_seq[0,:] -= self.mX.dot(beta_seq[1:,])

        return {'beta': beta0,
                'beta_seq': beta_seq[:,:count+1],
                'residuals': res,
                'robust': tau,
                'niter': count}
    

    #  Adaptive Huber in high dim
    def gd_highdim(self, s,  lr, T,  
                    tau=None, robust=5,
                    beta0=np.array([]),
                    standardize=True, adjust=True):
        '''
        Performs high-dimensional gradient descent using adaptive Huber loss.

        Parameters:
        ----------
        s : int, sparsity level of the solution (number of non-zero coefficients).
        lr : float, learning rate for gradient descent.
        T : int, maximum number of iterations.
        tau : float, initial robust scale for Huber loss (calculated dynamically if None).
        robust : float, robustness multiplier for Huber loss (default: 5).
        beta0 : ndarray, initial coefficients (default: zeros).
        standardize : bool, whether to standardize covariates.
        adjust : bool, whether to adjust coefficients to the original scale.
        '''
        if len(beta0) == 0:
            beta0 = np.zeros(self.X.shape[1])
            if self.itcp: beta0[0] = np.mean(self.Y)
        if standardize: X = self.X1
        else: X= self.X
        if self.itcp:
           X_1 = X[:,1:]
        n = X.shape[0]
    
        beta_seq = np.zeros([X.shape[1], T+1])
        beta_seq[:,0] = beta0
        
        res = self.Y-X @ beta0
        tau = mad(res)*robust  ## robust scale
        beta0_itcp = beta0[0] ## intercept
        beta0_rest = beta0[1:] ## rest
        count = 0
        
        while count < T: 
            beta0_itcp += lr*np.mean(huber_score(res, tau))
            beta0_rest += (lr/n) * (X_1.T @ huber_score(res, tau)) 
            beta0_rest =  ht(beta0_rest, s=s )
            beta0 = np.hstack((beta0_itcp, beta0_rest))
            res = self.Y-X @ beta0
            tau = mad(res)*robust
            beta_seq[:, count+1] = np.copy(beta0)
            count += 1

        if standardize and adjust:
            beta0[self.itcp:] = beta0[self.itcp:]/self.sdX
            beta_seq[self.itcp:,] = beta_seq[self.itcp:,]/self.sdX[:,None]
            if self.itcp: 
                beta0[0] -= self.mX.dot(beta0[1:])
                beta_seq[0,:] -= self.mX.dot(beta_seq[1:,])
            
 

        return  {'beta': beta0,
                'beta_seq': beta_seq[:,:count+1],
                'residuals': res,
                'robust': tau,
                'niter': count}
    

    def noisygd_highdim(self, s, lr, T,  mu, delta, robust_low1,robust_low2,  robust_high1=5, robust_high2=5,
                    beta0=np.array([]), B_high=1, 
                    standardize=True, adjust=True):
        '''
        Perform noisy gradient descent for high-dimensional sparse regression.
        
        Parameters:
            s : int, sparsity level for the regression coefficients.
            lr : float, learning rate for gradient updates.
            T : int, number of iterations.
            mu : float, privacy parameter controlling noise level.
            delta : float, privacy parameter controlling noise level.
            robust_low1 : float, robustness parameter for intercept(noiseless).
            robust_low2 : float, robustness parameter for intercept(noise).
            robust_high1 : float, robustness parameters for covariates(noiseless).
            robust_high2 : float, robustness parameters for covariates(noise).
            beta0 : ndarray, initial guess for the regression coefficients.
            B_low, B_high : float, clipping bounds for low and high.
            standardize : bool, whether to standardize the design matrix.
            adjust : bool, whether to adjust for mean and standard deviation in output. 
        '''
        if len(beta0) == 0:
            beta0 = np.zeros(self.X.shape[1])
            if self.itcp: beta0[0] = np.mean(self.Y)
        if standardize: X = self.X1
        else: X= self.X
        # Clip the non-intercept part of X if intercept is used
        if self.itcp:
            trun_X1 = clipping_inf(X[:, 1:], B_high)
            X_1 = X[:,1:]
        n = X.shape[0]
        # Initialize regression coefficients for two methods
        beta1 = np.copy(beta0)
        beta1_itcp = beta1[0]
        beta1_rest = beta1[1:]
        beta2 = np.copy(beta0)
        beta2_itcp = beta2[0]
        beta2_rest = beta2[1:] 
         # Initialize sequences to store coefficients at each iteration
        beta_seq1 = np.zeros([X.shape[1], T+1])
        beta_seq2 = np.zeros([X.shape[1], T+1]) 
        beta_seq1[:,0] = beta1
        beta_seq2[:,0] = beta2 
        # Compute initial residuals
        res1 = self.Y-X @ beta1
        res2 = self.Y-X @ beta2 
        # Compute robust scale parameters using median absolute deviation (MAD)
        tau_low1 = mad(res1)*robust_low1
        tau_low2 = mad(res1)*robust_low2
        tau_high1 = mad(res1)*robust_high1  ## robust scale 
        tau_high2 = mad(res2)*robust_high2    
 
        count = 0
        lambda_scale=2*(lr/n)*B_high*tau_high2
        while count < T: 
            diff1 = lr*  np.mean(huber_score(res1, tau_low1))  
            beta1_itcp += diff1 
            beta1_rest += (lr/n) * (X_1.T @ huber_score(res1, tau_high1))
            beta1_rest =  ht(beta1_rest, s=s )
            beta1 = np.hstack((beta1_itcp, beta1_rest))
            res1 = self.Y-X @ beta1 
            beta_seq1[:, count+1] = np.copy(beta1)
            
            diff2 = lr* (np.mean(huber_score(res2, tau_low2)) )\
                             + np.random.laplace(0, lambda_scale*2*np.sqrt(5* s*np.log(1/delta))/mu) 
            beta2_itcp += diff2
            beta2_rest += (lr/n) * (trun_X1.T @ huber_score(res2, tau_high2))
            beta2_rest =  noisyht(beta2_rest, s=s,mu=mu/T, delta=delta/T, lambda_scale=lambda_scale )
            beta2 = np.hstack((beta2_itcp, beta2_rest))
            res2 = self.Y-X @ beta2 
            beta_seq2[:, count+1] = np.copy(beta2) 

            count += 1

        if standardize and adjust:
            beta1[self.itcp:] = beta1[self.itcp:]/self.sdX
            beta_seq1[self.itcp:,] = beta_seq1[self.itcp:,]/self.sdX[:,None]
            beta2[self.itcp:] = beta2[self.itcp:]/self.sdX
            beta_seq2[self.itcp:,] = beta_seq2[self.itcp:,]/self.sdX[:,None] 
            if self.itcp: 
                beta1[0] -= self.mX.dot(beta1[1:])
                beta_seq1[0,:] -= self.mX.dot(beta_seq1[1:,])
                beta2[0] -= self.mX.dot(beta2[1:])
                beta_seq2[0,:] -= self.mX.dot(beta_seq2[1:,]) 
            
 

        return  {'beta1': beta1, 'beta2': beta2, 
                'beta_seq1': beta_seq1[:,:count+1],'beta_seq2': beta_seq2[:,:count+1], 
                'residuals1': res1,'residuals2': res2, 
                'niter': count}
    
 
    def noisygd_highdim_comp(self, s, lr, T,  mu, delta, robust_low,  robust_high1=5, robust_high2=5,
                beta0=np.array([]),  B_high=1, 
                standardize=True, adjust=True):
        ''' 
        Perform noisy gradient descent for high-dimensional sparse regression, 
        leaving the intercept noiseless for comparison with DP Ls.
        
        Parameters:
            s : int, sparsity level for the regression coefficients.
            lr : float, learning rate for gradient updates.
            T : int, number of iterations.
            mu : float, privacy parameter controlling noise level.
            delta : float, privacy parameter controlling noise level.
            robust_low : float, robustness parameter for intercept(noiseless).
            robust_high1 : float, robustness parameters for covariates(noiseless).
            robust_high2 : float, robustness parameters for covariates(noise).
            beta0 : ndarray, initial guess for the regression coefficients.
            B_high : float, clipping bound for high values.
            standardize : bool, whether to standardize the design matrix.
            adjust : bool, whether to adjust for mean and standard deviation in output. 
        '''
        if len(beta0) == 0:
            beta0 = np.zeros(self.X.shape[1])
            if self.itcp: beta0[0] = np.mean(self.Y)
        if standardize: X = self.X1
        else: X= self.X
        if self.itcp:
            trun_X1 = clipping_inf(X[:, 1:], B_high)
            X_1 = X[:,1:]
        n = X.shape[0]
        
        beta1 = np.copy(beta0)
        beta1_itcp = beta1[0]
        beta1_rest = beta1[1:]
        beta2 = np.copy(beta0)
        beta2_itcp = beta2[0]
        beta2_rest = beta2[1:] 

        beta_seq1 = np.zeros([X.shape[1], T+1])
        beta_seq2 = np.zeros([X.shape[1], T+1]) 
        beta_seq1[:,0] = beta1
        beta_seq2[:,0] = beta2 
        
        res1 = self.Y-X @ beta1
        res2 = self.Y-X @ beta2 
        tau_low = mad(res1)*robust_low
        tau_high1 = mad(res1)*robust_high1   
        tau_high2 = mad(res2)*robust_high2    
        count = 0
        
        while count < T: 
            diff1 = lr*  np.mean(huber_score(res1, tau_low))  
            beta1_itcp += diff1 
            beta1_rest += (lr/n) * (X_1.T @ huber_score(res1, tau_high1))
            beta1_rest =  ht(beta1_rest, s=s )
            beta1 = np.hstack((beta1_itcp, beta1_rest))
            res1 = self.Y-X @ beta1 
            beta_seq1[:, count+1] = np.copy(beta1)
            
            diff2 = lr* (np.mean(huber_score(res2, tau_low)) )
            beta2_itcp += diff2
            beta2_rest += (lr/n) * (trun_X1.T @ huber_score(res2, tau_high2))
            beta2_rest =  noisyht(beta2_rest, s=s,mu=mu/T, delta=delta/T, lambda_scale=2*(lr/n)*B_high*tau_high2 )
            beta2 = np.hstack((beta2_itcp, beta2_rest))
            res2 = self.Y-X @ beta2 
            beta_seq2[:, count+1] = np.copy(beta2)
            
            count += 1

        if standardize and adjust:
            beta1[self.itcp:] = beta1[self.itcp:]/self.sdX
            beta_seq1[self.itcp:,] = beta_seq1[self.itcp:,]/self.sdX[:,None]
            beta2[self.itcp:] = beta2[self.itcp:]/self.sdX
            beta_seq2[self.itcp:,] = beta_seq2[self.itcp:,]/self.sdX[:,None] 
            if self.itcp: 
                beta1[0] -= self.mX.dot(beta1[1:])
                beta_seq1[0,:] -= self.mX.dot(beta_seq1[1:,])
                beta2[0] -= self.mX.dot(beta2[1:])
                beta_seq2[0,:] -= self.mX.dot(beta_seq2[1:,]) 


        return  {'beta1': beta1, 'beta2': beta2, 
                'beta_seq1': beta_seq1[:,:count+1],'beta_seq2': beta_seq2[:,:count+1], 
                'residuals1': res1,'residuals2': res2, 
                'robust1': tau_high1,'robust2': tau_high2, 
                'niter': count}
     

    def noisygd_ls(self, s, lr, T, mu,  delta, 
                    C, R, 
                    beta0=np.array([]),  B=1, 
                    standardize=True, adjust=True):
        '''
        Perform differentially private linear regression using the algorithm 
        described by Cai, Wang, and Zhang (2021). 
        
        Parameters:
            s : int, sparsity level of the parameter vector.
            lr : float, learning rate for gradient updates.
            T : int, number of iterations.
            mu : float, privacy parameter controlling noise level.
            delta : float, privacy parameter controlling noise level.
            C : float, clipping bound for the coefficients.
            R : float, clipping bound for the response variable.
            beta0 : ndarray, initial estimate for the parameter vector.
            B : float, scaling factor.
            standardize : bool, whether to standardize the design matrix.
            adjust : bool, whether to adjust the final output for standardization.
        ''' 
        if len(beta0) == 0:
            beta0 = np.zeros(self.X.shape[1])
            if self.itcp: beta0[0] = np.mean(self.Y)
        if standardize: X = self.X1
        else: X= self.X 
        n = X.shape[0]
    
        beta_seq = np.zeros([X.shape[1], T+1])
        beta_seq[:,0] = beta0
        
        res_new = projection(self.Y,R)-X @ beta0
        count = 0
        
        while count < T:
            beta0 += (lr/n) * (X.T @ res_new)
            beta0 =  projection(noisyht(beta0, s=s, mu=mu/T, delta=delta/T, lambda_scale= (lr/n)*B),C)
            res_new = projection(self.Y,R)-X @ beta0 
            beta_seq[:, count+1] = np.copy(beta0)
            count += 1

        if standardize and adjust:
            beta0[self.itcp:] = beta0[self.itcp:]/self.sdX
            beta_seq[self.itcp:,] = beta_seq[self.itcp:,]/self.sdX[:,None]
            if self.itcp: 
                beta0[0] -= self.mX.dot(beta0[1:])
                beta_seq[0,:] -= self.mX.dot(beta_seq[1:,])
            
        return  {'beta': beta0,
                'beta_seq': beta_seq[:,:count+1],
                'residuals': res_new, 
                'niter': count} 
    

 
