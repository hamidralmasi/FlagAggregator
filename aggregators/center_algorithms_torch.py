'''
This file contains the PyTorch implementation of FlagIRLS, l2-median and grassmannian gradient descent algorithms.

by Hamidreza Almasi
'''
import numpy as np
import torch
import sys

def gr_log(X,Y):
    '''
    Log map on the Grassmannian.
    
    Inputs:
        X (np.array) a point about which the tangent space has been computed
        Y (np.array) the point on the Grassmannian manifold that's mapped to the tangent space of X
    Outputs:
        TY (np.array) Y in the tangent space of X
    '''
    m = X.shape[0]

    #temp = (np.eye(m)-X @ X.T) @ Y @ np.linalg.inv(X.T@Y)
    #The following line is a slightly faster way to compute temp.

    temp = np.eye(m) @ Y @ np.linalg.inv(X.T @ Y) - X @ (X.T @ Y) @ np.linalg.inv(X.T @ Y)
    U,S,V = np.linalg.svd(temp, full_matrices = False)
    Theta = np.arctan(S)
    
    TY = U @ np.diag(Theta) @ V.T
    
    return TY
                                             

def gr_exp(X, TY):
    '''
    Exponential map on the Grassmannian.

    Inputs:
        X: (np.array) is the point about which the tangent space has been
          computed.
        TY: (np.array) is a point in the tangent space of X.
    Outputs:
        Y: The output of the exponential map.
    
    '''
    
    U, S, V = np.linalg.svd(TY, full_matrices = False)
    Y = X @ V @ np.diag(np.cos(S)) + U @ np.diag(np.sin(S))

    return Y



def gr_dist(X, Y):
    '''
    Geodesic distance on the Grassmannian

    inputs:
        X- numpy array
        Y- numpy array
    outputs:
        dist- the geodesic distance between X and Y
    '''
    if X.shape[1] > 1:
        U,S,V = np.linalg.svd(X.T @ Y, full_matrices = False)
        S[np.where(S >1)] = 1
#         S[np.where(S < -1)] = -1
        angles = np.real(np.arccos(S))
#         print(angles)
        dist = np.linalg.norm(angles)
    else:
        dist = calc_error_1_2([X], Y, 'geodesic')
    return dist


def l2_median(data, alpha, r, max_itrs, seed=0, init_datapoint = False):
    '''
    Code adopted from Tim Marrinan (translated from matlab into python)

    inputs:
        data- list of numpy arrays 
        alpha- float for the step size
        r- integer for Gr(r,n) where the output 'lives'
        max_itrs- integer for the maximum number of iterations
        seed- integer for the numpy random seed for the algorithm initialization
        init_datapoint- boolean, True for intializing at a datapoint, False for random initialization
    outputs:
        Y- numpy array for the l2-median
        err- objective function values at each iteration
    '''
    
    n = data[0].shape[0]
    
    if init_datapoint:
        np.random.seed(seed)
        Y = data[np.random.randint(len(data))]
    else:
        np.random.seed(seed)
        Y_raw = np.random.rand(n,r)-.5
        Y = np.linalg.qr(Y_raw)[0][:,:r]
    
    itr = 0
    errs = []
    diff = 1
    
    while diff > 0.000001 and itr < max_itrs:
        d_fracs = 0
        ld_fracs = np.empty((n,r))
        dists = []
        for x in data:
            dists.append(gr_dist(x, Y))
            if dists[-1] > .0001:
                d_fracs += 1 / dists[-1]
                ld_fracs += gr_log(Y, x) / dists[-1]
            else:
                print('converged to datapoint')

        if len(ld_fracs)==0:
            return Y
        else:
            vk = ld_fracs/d_fracs
            Y = gr_exp(Y, alpha * vk)
            
            errs.append(np.sum(dists))
            
            if itr > 0:
                diff = np.abs(errs[-2] - errs[-1])
            
            if not np.allclose(Y.T @ Y, np.eye(r,r)):
                Y = np.linalg.qr(Y)[0][:,:r]
            
            itr+=1 
    
    return Y, errs



def calc_error_1_2(data, Y, sin_cos):
    '''
    Calculate objective function value. 

    Inputs:
        data - a list of numpy arrays representing points in Gr(k_i,n)
        Y - a numpy array representing a point on Gr(r,n) 
        sin_cos - a string defining the objective function
                    'cosine' = Maximum Cosine
                    'sine' = Sine Median
                    'sinsq' = Flag Mean
                    'geodesic' = Geodesic Median (CAUTION: only for k_i = r = 1)
    Outputs:
        err - objective function value
    '''
    k = Y.shape[1]
    # print("k: ", k)
    err = torch.zeros(1).cuda()
    if sin_cos == 'sine':
        for x in data:
            r = np.min([k,x.shape[1]])
            sin_sq = r - torch.trace(Y.T @ x @ x.T @ Y)

            #fixes numerical errors
            if sin_sq < 0:
                sin_sq = torch.zeros(1).cuda()
            err += torch.sqrt(sin_sq)
    
    elif sin_cos == 'sinesq':
        for x in data:
            r = np.min([k,x.shape[1]])
            sin_sq = r - torch.trace(Y.T @ x @ x.T @ Y)
            #fixes numerical errors
            if sin_sq < 0:
                sin_sq = 0
            err += sin_sq
    elif sin_cos == 'geodesic':
        for x in data:
            cos = (Y.T @ x @ x.T @ Y)[0][0]
            #fixes numerical errors
            if cos > 1:
                cos = 1
            elif cos < 0:
                cos = 0
            err += torch.arccos(torch.sqrt(cos))
    elif sin_cos == 'l2_med':
        for x in data:
            err += gr_dist(x, Y)
    return err



def flag_mean(data, r):
    '''
    Calculate the Flag Mean

    Inputs:
        data - list of numpy arrays representing points on Gr(k_i,n)
        r - integer number of columns in flag mean
    Outputs:
        mean - a numpy array representing the Flag Mean of the data
    '''
    X = torch.hstack(data)

    # print("Shape of X:" + str(np.shape(X)))

    # print("r is: " + str(r))

    # print("Length of data: " + str(len(data)))
    # mean = torch.svd(X)[0][:,:r]
    mean = torch.linalg.svd(X, full_matrices = False)[0][:,:r]
    return mean




def flag_mean_iteration(data, Y0, weight, itr, eps = .0000001, **kwargs):
    '''
    Calculates a weighted Flag Mean of data using a weight method for FlagIRLS
    eps = .0000001 for paper examples

    Inputs:
        data - list of numpy arrays representing points on Gr(k_i,n)
        Y0 - a numpy array representing a point on Gr(r,n)
        weight - a string defining the objective function
                    'sine' = flag median
                    'sinsq' = flag mean
        eps - a small perturbation to the weights to avoid dividing by zero
    Outputs:
        Y- the weighted flag mean
    '''
    r = Y0.shape[1]
    
    aX = []
    al = []

    ii=0
    log_file = kwargs.get('log_file', None)
    data_itr = kwargs.get('data_itr', 0)
    lambda_ = kwargs.get('lambda_', 0)
    if log_file is not None:
        log_file.write("data_itr," + str(data_itr) + "," + "irls_itr," + str(itr) + ",")
    p = len(data)
    for x in data:
        if weight == 'sine':
            m = np.min([r,x.shape[1]])
            sinsq = torch.absolute((m - torch.trace(Y0.T @ x @ x.T @ Y0)))
            # print("sinesq: "+ str(sinsq))
            w = (sinsq+eps)**(-1/4)
            if(log_file):
                log_file.write(str(w.item()) + ",")
            al.append(w) #(1/p)*# 
        else:
            print('unrecognized weight')
        # print("x is: " + str(x))
        # print("is cuda al[-1]*x: " + str((al[-1]*x).is_cuda))
        aX.append(al[-1]*x)
        ii+= 1

    # print("ii: " + str(ii))
    # print("len(aX): " + str(len(aX)))

    for x in data:
        for y in data:
            if x is not y:
                if weight == 'sine':
                    m = np.min([r,x.shape[1]])
                    sinsq = torch.absolute((m - torch.trace(Y0.T @ (x - y) @ (x - y).T @ Y0)))
                    w = ((sinsq+eps)**(-1/4)* lambda_ / (p*(p-1)))
                    if(log_file):
                        log_file.write(str(w.item()) + ",")
                    al.append(w)
                else:
                    print('unrecognized weight')
                aX.append(al[-1]*(x - y))
    # aX = [aX[i] * (lambda_/(p*(p-1))) for i in range(ii - (p * (p-1)) , len(aX))]

    # # pick 2 * p distinct random pairs of indices from 0 to p-1
    # rand_pairs = np.random.choice(p, size = (2 * p, 2), replace = True)
    # cnt = 0
    # for k in range(len(rand_pairs)):
    #     i = rand_pairs[k][0]
    #     j = rand_pairs[k][1]
    #     if i == j:
    #         continue
    #     cnt += 1
    #     xi = data[i]
    #     xj = data[j]
    #     if weight == 'sine':
    #         m = np.min([r,xi.shape[1]])
    #         sinsq = torch.absolute((m - torch.trace(Y0.T @ (xi - xj) @ (xi - xj).T @ Y0)))
    #         w = (sinsq+eps)**(-1/4)
    #         if(log_file):
    #             log_file.write(str(w.item()) + ",")
    #         al.append(w)
    #     else:
    #         print('unrecognized weight')
    #     aX.append(al[-1]*(xi - xj))
    #     ii+= 1
    # # multiply elements of aX by lambda_/cnt
    # # print("len(aX): " + str(len(aX)))
    # # print("cnt: " + str(cnt))
    # aX = [aX[i] * (lambda_/cnt) for i in range(ii - cnt, len(aX))]
    

    # for i in range(num_rand):
    #     for j in range(num_rand):
    #         # pick a random index from the data
    #         idx = np.random.randint(0, p)
    #         # pick another random index from the data that is not the same as the first
    #         idx2 = np.random.randint(0, p)
    #         while idx2 == idx:
    #             idx2 = np.random.randint(0, p)
    #         # 
    #         if i != j:
    #             xi = data[i]
    #             xj = data[j]
    #             if weight == 'sine':
    #                 m = np.min([r,xi.shape[1]])
    #                 sinsq = torch.absolute((m - torch.trace(Y0.T @ (xi - xj) @ (xi - xj).T @ Y0)))
    #                 w = (sinsq+eps)**(-1/4)
    #                 if(log_file):
    #                     log_file.write(str(w.item()) + ",")
    #                 al.append((lambda_ / (p * (p - 1)))*w)
    #             else:
    #                 print('unrecognized weight')
    #             aX.append(al[-1]*(xi - xj))
    #             ii+= 1
        

    # print("al: " + str(al))
    # print("Shape of aX: " + str(np.shape(aX)))
    Y = flag_mean(aX, r)
    if(log_file):
        log_file.write("\n")
    return Y



def irls_flag(data, r, n_its, sin_cos, opt_err = 'geodesic', init = 'random', seed = 0, **kwargs): 
    '''
    Use FlagIRLS on data to output a representative for a point in Gr(r,n) 
    which solves the input objection function

    Repeats until iterations = n_its or until objective function values of consecutive
    iterates are within 0.0000000001 and are decreasing for every algorithm (except increasing for maximum cosine)

    Inputs:
        data - list of numpy arrays representing points on Gr(k_i,n)
        r - the number of columns in the output
        n_its - number of iterations for the algorithm
        sin_cos - a string defining the objective function for FlagIRLS
                    'sine' = flag median
        opt_err - string for objective function values in err (same options as sin_cos)
        init - string 'random' for random initlalization. 
            otherwise input a numpy array for the inital point
        seed - seed for random initialization, for reproducibility of results
    Outputs:
        Y - a numpy array representing the solution to the chosen optimization algorithm
        err - a list of the objective function values at each iteration (objective function chosen by opt_err)
    '''
    err = []
    n = data[0].shape[0]
    # print("n = ",n)

    #initialize
    if init == 'random':
        #randomly
        # torch random seed
        torch.manual_seed(seed)
        # create random (n, r)
        Y_raw = torch.rand(n, r) - 0.5
        # qr decomposition
        Y = torch.qr(Y_raw)[0][:,:r]
        Y = Y.to(device='cuda')
        
        # print("Y shape: " + str(Y.shape))
    elif init == 'data':
        torch.manual_seed(seed)
        Y = data[torch.randint(len(data))]
    else:
        Y = init

    # print("Shape of initial Y: " + str(np.shape(Y)))
    err.append(calc_error_1_2(data, Y, opt_err))
    
    #flag mean iteration function
    #uncomment the commented lines and 
    #comment others to change convergence criteria
    
    itr = 1
    diff = 1
    while itr <= n_its and diff > 0.0000000001 and err[itr - 1] > 0.0000000001:
        Y0 = Y.clone()

        Y = flag_mean_iteration(data, Y, sin_cos, itr, **kwargs)

        err.append(calc_error_1_2(data, Y, opt_err))
        diff  = err[itr-1] - err[itr]
           
        itr+=1
        # print("itr: " + str(itr))
        # print("err: " + str(err))
        # print("diff: "  + str(diff))

    if diff > 0:
        return Y, err
    else:
        return Y0, err[:-1]


def calc_gradient(data, Y0, weight = 'sine'):
    '''
    Calculates the gradient of a given Y0 and data given an objective function
    Inputs:
        data - list of numpy arrays representing points on Gr(k_i,n)
        Y0 - a representative for a point on Gr(r,n)
        weight - a string defining the objective function
                    'sine' = flag median
    Output:
        grad - numpy array of the gradient

    '''
    k = Y0.shape[1]
    aX = []
    al = []
    for x in data:
        if weight == 'sine':
            r = np.min([k,x.shape[1]])
            sin_sq = r - np.trace(Y0.T @ x @ x.T @ Y0)
            if sin_sq < .00000001 :
                sin_sq = 0
                print('converged to datapoint')
            else:
                al.append(sin_sq**(-1/4))
        else:
            print('weight must be sine')
        aX.append(al[-1]*x)

    big_X = np.hstack(aX)
    
    grad = big_X @ big_X.T @ Y0

    return grad



def gradient_descent(data, r, alpha, n_its, sin_cos, init = 'random', seed = 0):
    '''
    Runs Grassmannian gradient descent
    Inputs:
        data - list of numpy arrays representing points on Gr(k,n)
        r - integer for the number of columns in the output
        alpha - step size
        n_its - number of iterations
        sin_cos - a string defining the objective function
                    'sine' = flag median
                    'sinsq' = flag mean
        init - string 'random' for random initlalization. 
            otherwise input a numpy array for the inital point
    Outputs:
        Y - a numpy array representing the solution to the chosen optimization algorithm
        err - a list of the objective function values at each iteration (objective function chosen by opt_err)
    '''
    n = data[0].shape[0]

    #initialize
    if init == 'random':
        np.random.seed(seed)
        #randomly
        Y_raw = np.random.rand(n,r)
        Y = np.linalg.qr(Y_raw)[0][:,:r]
    else:
        Y = init

    err = []
    err.append(calc_error_1_2(data, Y, sin_cos))

    for _ in range(n_its):
        Fy = calc_gradient(data,Y,sin_cos)
        # project the gradient onto the tangent space
        G = (np.eye(n)-Y@Y.T)@Fy
        
        [U,S,V] = np.linalg.svd(G)
        cosin = np.diag(np.cos(-alpha*S))
        sin = np.vstack([np.diag(np.sin(-alpha*S)), np.zeros((n-r,r))])
        if cosin.shape[0] == 1:
            Y = Y*V*cosin*V.T+U@sin *V.T
        else:
            Y = Y@V@cosin@V.T+U@sin@V.T
        
        err.append(calc_error_1_2(data, Y, sin_cos))
    return Y, err