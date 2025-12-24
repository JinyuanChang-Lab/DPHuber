import numpy as np 

def w_gamma(u, gamma):
    return np.minimum(u / gamma, 1.0)

def psi_tau_new(u, tau): 
    return np.minimum((np.abs(u))**2, tau**2)  

def gene_E(p):   
    E = np.zeros((p, p)) 
    triu_indices = np.triu_indices(p)
    upper_values = np.random.randn(len(triu_indices[0])) 
    E[triu_indices] = upper_values 
    E = E + np.triu(E, 1).T  
    return E
 
##### method  : \psi_{\tau}^2(\varepsilon) \sim \varepsilon^2

def hatXi(X, Y,beta,tau_new):
    n = X.shape[0] 
    weighted_outer_products = np.einsum('ni,nj->nij', X, X)  #  X_i X_i^T shape (n, p, p)
    #hat_Sigma
    Sigma_hat = np.sum(weighted_outer_products, axis=0) * (1 / n) # shape ( p, p)    
    #hat_Omega 
    res = Y - X @ beta
    psi_vals = psi_tau_new(res, tau_new) # shape (n,)
    total_weights = psi_vals  / n  # shape (n,)
    Omega_hat = np.tensordot(total_weights, weighted_outer_products, axes=([0], [0]))  # shape ( p, p)
    #hat_Xi
    Sigma_inv = np.linalg.inv(Sigma_hat) 
    Xi_tilde = Sigma_inv @ Omega_hat @ Sigma_inv     
    return  Xi_tilde


def hatXi_DP_priv(X, Y, beta,gamma, tau_new,epsilon,delta):
    zeta=1e-4
    n,p = X.shape 
    E = gene_E(p)
    weights = w_gamma(np.sum(X**2, axis=1), gamma**2)  # shape (n,)
    weighted_outer_products = np.einsum('ni,nj->nij', X, X)  # shape (n, p, p)
    #hat_Sigma_gamma
    epsilon_sigma = epsilon/2
    delta_sigma = delta/2
    Sigma_hat_pre = np.tensordot(weights / n, weighted_outer_products, axes=([0], [0])) # shape ( p, p)   
    Sigma_hat_pre_priv = Sigma_hat_pre+2*(gamma**2)*((2*np.log(1.25/delta_sigma))**0.5)/(n*epsilon_sigma)*E
    Sigma_hat_sym = (Sigma_hat_pre_priv + Sigma_hat_pre_priv.T) / 2
    eigvals_1, eigvecs_1 = np.linalg.eigh(Sigma_hat_sym) # eigenvaule decomposition  
    eigvals_clipped_1 = np.maximum(eigvals_1, zeta) # adjust eigenvaule 
    Sigma_hat_proj = eigvecs_1 @ np.diag(eigvals_clipped_1) @ eigvecs_1.T # matrix reconstructed # shape ( p, p)   
    #hat_Omega 
    epsilon_omega = epsilon/2
    delta_omega = delta/2
    res = Y - X @ beta
    psi_vals = psi_tau_new(res, tau_new)  # shape (n,)
    total_weights = psi_vals * weights / n  # shape (n,)
    Omega_hat_pre = np.tensordot(total_weights, weighted_outer_products, axes=([0], [0]))  # shape ( p, p)
    Omega_hat_pre_priv = Omega_hat_pre+2*(gamma**2)*(tau_new**2)*((2*np.log(1.25/delta_omega))**0.5)/(n*epsilon_omega)*E
    Omega_hat_sym = (Omega_hat_pre_priv + Omega_hat_pre_priv.T) / 2
    eigvals_2, eigvecs_2 = np.linalg.eigh(Omega_hat_sym) # eigenvaule decomposition  
    eigvals_clipped_2 = np.maximum(eigvals_2, zeta) # adjust eigenvaule 
    Omega_hat_proj = eigvecs_2 @ np.diag(eigvals_clipped_2) @ eigvecs_2.T # matrix reconstructed # shape ( p, p)   
    #hat_Xi
    Sigma_inv = np.linalg.inv(Sigma_hat_proj) 
    Xi_tilde = Sigma_inv @ Omega_hat_proj @ Sigma_inv   
    return  Xi_tilde


def hatXi_GDP_priv(X, Y,beta, gamma, tau_new,epsilon):
    zeta=1e-4
    n,p = X.shape 
    E = gene_E(p)
    weights = w_gamma(np.sum(X**2, axis=1), gamma**2)  # shape (n,)
    weighted_outer_products = np.einsum('ni,nj->nij', X, X)  # shape (n, p, p)
    #hat_Sigma_gamma
    epsilon_sigma = epsilon/2
    Sigma_hat_pre = np.tensordot(weights / n, weighted_outer_products, axes=([0], [0])) # shape ( p, p)   
    Sigma_hat_pre_priv = Sigma_hat_pre+2*(gamma**2)/(n*epsilon_sigma)*E
    Sigma_hat_sym = (Sigma_hat_pre_priv + Sigma_hat_pre_priv.T) / 2
    eigvals_1, eigvecs_1 = np.linalg.eigh(Sigma_hat_sym) # eigenvaule decomposition  
    eigvals_clipped_1 = np.maximum(eigvals_1, zeta) # adjust eigenvaule 
    Sigma_hat_proj = eigvecs_1 @ np.diag(eigvals_clipped_1) @ eigvecs_1.T # matrix reconstructed # shape ( p, p)   
    #hat_Omega 
    epsilon_omega = epsilon/2
    res = Y - X @ beta
    psi_vals = psi_tau_new(res, tau_new)  # shape (n,)
    total_weights = psi_vals * weights / n  # shape (n,)
    Omega_hat_pre = np.tensordot(total_weights, weighted_outer_products, axes=([0], [0]))  # shape ( p, p)
    Omega_hat_pre_priv = Omega_hat_pre+2*(gamma**2)*(tau_new**2)/(n*epsilon_omega)*E
    Omega_hat_sym = (Omega_hat_pre_priv + Omega_hat_pre_priv.T) / 2
    eigvals_2, eigvecs_2 = np.linalg.eigh(Omega_hat_sym) # eigenvaule decomposition  
    eigvals_clipped_2 = np.maximum(eigvals_2, zeta) # adjust eigenvaule 
    Omega_hat_proj = eigvecs_2 @ np.diag(eigvals_clipped_2) @ eigvecs_2.T # matrix reconstructed # shape ( p, p)   
    #hat_Xi
    Sigma_inv = np.linalg.inv(Sigma_hat_proj) 
    Xi_tilde = Sigma_inv @ Omega_hat_proj @ Sigma_inv   
    return  Xi_tilde

 
 