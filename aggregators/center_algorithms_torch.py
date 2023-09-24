
import numpy as np
import torch
import sys
import csv

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
    if torch.cuda.is_available():
        err = torch.zeros(1).cuda()
    else:
        err = torch.zeros(1)
    if sin_cos == 'sine':
        for x in data:
            r = np.min([k,x.shape[1]])
            sin_sq = r - torch.trace(Y.T @ x @ x.T @ Y)

            #fixes numerical errors
            if sin_sq < 0:
                if torch.cuda.is_available():
                    sin_sq = torch.zeros(1).cuda()
                else:
                    sin_sq = torch.zeros(1)
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
            w = (sinsq+eps)**(-1/4)
            if(log_file):
                log_file.write(str(w.item()) + ",")
            al.append(w) #(1/p)*#
        else:
            print('unrecognized weight')

        aX.append(al[-1]*x)
        ii+= 1


    # if lambda is positive:
    if lambda_ > 0:
        # begin computing pairwise distances:
        for x in data:
            for y in data:
                if x is not y:
                    if weight == 'sine':
                        m = np.min([r,x.shape[1]])
                        sinsq = torch.absolute((m - torch.trace(Y0.T @ (x - y) @ (x - y).T @ Y0)))
                        w = ((sinsq+eps)**(-1/4)) * lambda_ / (p*(p-1))
                        if(log_file):
                            log_file.write(str(w.item()) + ",")
                        al.append(w)
                    else:
                        print('unrecognized weight')
                    aX.append(al[-1]*(x - y))
        # end computing pairwise distances

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

        Y_raw = torch.rand(n, r) - 0.5
        if torch.cuda.is_available():
            Y_raw = Y_raw.to(device='cuda')


        Y_mean =torch.sum(torch.stack(data), dim=0).div(len(data))
        norm = torch.norm(Y_mean)

# Normalize the tensor by dividing each element by the norm
        Y_mean = Y_mean / norm
        if torch.cuda.is_available():
            Y_mean = Y_mean.to(device='cuda')
        iter = kwargs.get('iter', 0)
        values = [(data[i].T@Y_mean@Y_mean.T@data[i])/(data[i].T@data[i]) for i in range(len(data))]

        header = ['w' + str(i+1) for i in range(15)]
        with open('/home/data/Garfield/pytorch_impl/applications/Aggregathor/values_suboptimal.csv', 'a+', newline='') as f:
            writer = csv.writer(f)
            if iter == 1:
                f.truncate(0)
                writer.writerow(header)
            else:
                values_np = [values[i].cpu().numpy().flatten().tolist() for i in range(len(values))]
                values_np = [str(val).strip('[]') for val in values_np]
                writer.writerow(values_np)

        # qr decomposition
        Y = torch.qr(Y_raw)[0][:,:r]
        if torch.cuda.is_available():
            Y = Y.to(device='cuda')



    elif init == 'data':
        torch.manual_seed(seed)
        Y = data[torch.randint(len(data))]
    else:
        Y = init

    err.append(calc_error_1_2(data, Y, opt_err))

    #flag mean iteration function


    itr = 1
    diff = 1
    while itr <= n_its and diff > 0.0000000001 and err[itr - 1] > 0.0000000001:
        Y0 = Y.clone()

        Y = flag_mean_iteration(data, Y, sin_cos, itr, **kwargs)

        err.append(calc_error_1_2(data, Y, opt_err))
        diff  = err[itr-1] - err[itr]

        itr+=1


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
