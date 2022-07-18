import torch

def _mix_rbf_kernel(X, Y, sigma_list):

    m = X.size(0)
    # n = Y.size(0)

    Z = torch.cat((X, Y), 0)
    ZZT = torch.mm(Z, Z.t())
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()

    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma**2)
        K += torch.exp(-gamma*exponent)

    return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)


def _mmd2(K_XX, K_XY, K_YY, biased=True):

    m = K_XX.size(0)
    n = K_YY.size(0)

    diag_X = torch.diag(K_XX)
    diag_Y = torch.diag(K_YY)
    sum_diag_X = torch.sum(diag_X)
    sum_diag_Y = torch.sum(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    if biased:
        mmd2 = (Kt_XX_sum + sum_diag_X) / (m*m) + (Kt_YY_sum + sum_diag_Y) / (n*n) - 2.0 * K_XY_sum /(m*n)
    else:
        mmd2 = Kt_XX_sum / (m*(m-1)) + Kt_YY_sum / (n*(n-1)) - 2.0 * K_XY_sum / (m*n)

    return mmd2


def _wmmd2(K_XX, K_XY, K_YY, W_X, W_Y, biased=True):

    m = K_XX.size(0)
    n = K_YY.size(0)

    diag_W_X = torch.diag(W_X.float())
    K_XX_W = torch.mm(torch.mm(diag_W_X, K_XX), diag_W_X)
    diag_W_Y = torch.diag(W_Y.float())
        
    K_YY_W = torch.mm(torch.mm(diag_W_Y, K_YY), diag_W_Y)
    K_XY_W = torch.mm(torch.mm(diag_W_X, K_XY), diag_W_Y)


    diag_X_W = torch.diag(K_XX_W)
    diag_Y_W = torch.diag(K_YY_W)
    sum_diag_X_W = torch.sum(diag_X_W)
    sum_diag_Y_W = torch.sum(diag_Y_W)

    Kt_XX_W_sums = K_XX_W.sum(dim=1) - diag_X_W
    Kt_YY_W_sums = K_YY_W.sum(dim=1) - diag_Y_W
    K_XY_W_sums_0 = K_XY_W.sum(dim=0)

    Kt_XX_W_sum = Kt_XX_W_sums.sum()
    Kt_YY_W_sum = Kt_YY_W_sums.sum()
    K_XY_W_sum = K_XY_W_sums_0.sum()

    W_X_sum = W_X.sum()
    W_X_2_sum = torch.sum(W_X**2)
    W_Y_sum = W_Y.sum()
    W_Y_2_sum = torch.sum(W_Y**2)

    if biased:
        wmmd2 = (Kt_XX_W_sum + sum_diag_X_W) / (W_X_sum*W_X_sum) + (Kt_YY_W_sum + sum_diag_Y_W) / (W_Y_sum*W_Y_sum) - 2.0 * K_XY_W_sum / (W_X_sum*W_Y_sum)
    else:
        wmmd2 = Kt_XX_W_sum / (W_X_sum*W_X_sum - W_X_2_sum) + Kt_YY_W_sum / (W_Y_sum*W_Y_sum - W_Y_2_sum) - 2.0 * K_XY_W_sum / (W_X_sum*W_Y_sum)

    return wmmd2


def mix_rbf_mmd2(X, Y, sigma_list, biased=True):

    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    return _mmd2(K_XX, K_XY, K_YY, biased=biased)


def mix_rbf_wmmd2(X, Y, W_X, W_Y, sigma_list, biased=True):

    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    return _wmmd2(K_XX, K_XY, K_YY, W_X, W_Y, biased=biased)


def mix_rbf_jmmd2(X_1, X_2, Y_1, Y_2, sigma_list_1, sigma_list_2, biased=True):
    
    K_XX_1, K_XY_1, K_YY_1, d_1 = _mix_rbf_kernel(X_1, Y_1, sigma_list_1)
    K_XX_2, K_XY_2, K_YY_2, d_2 = _mix_rbf_kernel(X_2, Y_2, sigma_list_2)
    
    K_XX = torch.mul(K_XX_1, K_XX_2)
    K_XY = torch.mul(K_XY_1, K_XY_2)
    K_YY = torch.mul(K_YY_1, K_YY_2)
    
    return _mmd2(K_XX, K_XY, K_YY, biased=biased)


def mix_rbf_jwmmd2(X_1, X_2, Y_1, Y_2, W_X, W_Y, sigma_list_1, sigma_list_2, biased=True):
    
    K_XX_1, K_XY_1, K_YY_1, d_1 = _mix_rbf_kernel(X_1, Y_1, sigma_list_1)
    K_XX_2, K_XY_2, K_YY_2, d_2 = _mix_rbf_kernel(X_2, Y_2, sigma_list_2)
    
    K_XX = torch.mul(K_XX_1, K_XX_2)
    K_XY = torch.mul(K_XY_1, K_XY_2)
    K_YY = torch.mul(K_YY_1, K_YY_2)
    
    return _wmmd2(K_XX, K_XY, K_YY, W_X, W_Y, biased=biased)













