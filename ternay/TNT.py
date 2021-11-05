# coding=utf-8
import torch
def normalize_row(target_vector):
    """
    convert each filter in a kernel to a unit vector
    ----------------------------
    Input

    target_vector： tensor from a kernel
    ----------------------------
    Output

    target_hat: the unit vector
    """

    target_hat = target_vector / (torch.norm(target_vector, p=2, dim=-1, keepdim=True) + 1e-6)

    return target_hat


def similar_cos(w_flatten_f):
    """
    calculate the similarity between ternary vector and its corresponding original vector
    ------------------------------
    Input

    w_flatten_f: vector original vector
    t_dim: dimension of a kernel
    ------------------------------
    Output

    target_hat_sorted_index: index of an original vector in sorted form (descending sort)
    similar_value: list cosine similarity of
    maxValue_conv: the maximum cosine similarity between ternary and original
    maxIndex_conv: the index of the ternary vector
    """

    # calculating the cosine similarity
    target_vector = w_flatten_f  # not need?
    target_hat = normalize_row(target_vector)
    target_hat_sorted, target_hat_sorted_index = torch.sort(torch.abs(target_hat),
                                                            descending=True)  # descending sort last dim

    temp = torch.cumsum(target_hat_sorted, -1)  # cumsum last dim
    similar_value = temp / torch.sqrt(torch.arange(1, temp.size(-1) + 1).to(temp.device).float())

    # similar_value = similar_value[:, 1:]
    maxValue_conv, maxIndex_conv = torch.max(similar_value, dim=-1, keepdim=True)

    return target_hat_sorted_index, similar_value, maxValue_conv, maxIndex_conv


def order_vec(kernel_flatten, target_hat_sorted_index, maxIndex_conv):
    """
    convert sorted ternary vector to original order
    ----------------------
    Parameters
    ----------------------
    Inputs

    kernel_flatten: original vector
    target_hat_sorted_index: sorted ternary vector (decreasing order of original vector)
    maxIndex_conv: the ternary vector with largest cosine similarity
    -----------------------
    Outputs

    ternary_vector: ternary vectory with the order of origin.
    """

    index_tune = torch.clamp(maxIndex_conv, min=0, max=kernel_flatten.size(1))
    
    x_sorted = torch.gather(kernel_flatten, -1, target_hat_sorted_index)  # (N, F)
    x_abs_max = torch.gather(x_sorted, -1, index_tune).abs()  # (N, 1)
    ternary_vector = torch.sign(kernel_flatten)
    mask = torch.abs(kernel_flatten) < x_abs_max
    ternary_vector[mask] = 0
        
    return ternary_vector


# TNT convert
def TNT_convert(weights_f):
    """
    convert weights(vector) to ternary vector
    ---------------------
    Inputs

    weights_f: input tensor with floating type  (it is a matrix)
    ---------------------
    Outputs

    ternary_weights: ternary tensor
    maxValue_conv: the maximum cosine similarity
    """

    target_hat_sorted_index, similar_value, maxValue_conv, maxIndex_conv = similar_cos(weights_f)
    ternary_weights = order_vec(weights_f, target_hat_sorted_index, maxIndex_conv)

    return ternary_weights, maxValue_conv


# scaling
def scaling1(kernel_flatten, ternary_vector):
    
    floating_norm = torch.norm(kernel_flatten, p=2, dim=-1, keepdim=True)
    ternary_norm = torch.norm(ternary_vector, p=2, dim=-1, keepdim=True)
    scale_num = floating_norm / (ternary_norm + 1e-6)
    weights_t = scale_num * ternary_vector
    
    t_norm = torch.norm(weights_t, p=2, dim=-1, keepdim=True)
#     tt = (t_norm == floating_norm)
#     print('jklsjkljklfnl',tt[:10])
#     print('kkkkkk', t_norm[:10])
#     print('lllllll', floating_norm[:10])
#     stop()
    
    return weights_t

def scaling(kernel_flatten, ternary_vector):
    """
    Calculating a scaling number to reduce the error between ternary vector
    and its corresponding floating vector
    -----------------------
    Input

    kernel_flatten: floating vector
    ternary_vector: ternary tensor
    t_dim: dimension of the floating vector
    -----------------------
    Output

    weights_t: scaled ternary tensor with positive and negative scaling number
    """

    a = kernel_flatten
    t = ternary_vector

    ap = a.clone()
    an = a.clone()
    tp = t.clone()
    tn = t.clone()

    ap[a < 0.] = 0.
    an[a > 0.] = 0.
    tp[t < 0.] = 0.
    tn[t > 0.] = 0.

    pos_sum_zero_mask = torch.sum(ap, dim=-1, keepdim=True) != 0
    neg_sum_zero_mask = torch.sum(an, dim=-1, keepdim=True) != 0
    rp = torch.norm(ap, 2, dim=-1, keepdim=True) / (torch.norm(tp, dim=-1, keepdim=True) + 0.00001)
    rn = torch.norm(an, 2, dim=-1, keepdim=True) / (torch.norm(tn, dim=-1, keepdim=True) + 0.00001)

    rp = rp * torch.cosine_similarity(ap, tp, dim=-1).unsqueeze(-1)
    rp = rp * pos_sum_zero_mask

    rn = rn * torch.cosine_similarity(an, tn, dim=-1).unsqueeze(-1)
    rn = rn * neg_sum_zero_mask

    weights_t = tp * rp + tn * rn
    return weights_t


def kernels_cluster(weights_f, channel=False):
    """
    Output a scaled ternary tensor
    --------------------
    Inputs

    weights_f: tensor, kernels or filters of a network with floating type.
    channel: slicing a tensor along the channel fiber or along the frontal slice.
    ---------------------
    Output

    ternary_weights: ternary tensors without scaled.
    weights_t: ternary tensors with the same shape with input weights and scaled.
    """

    t_dim = len(weights_f.size())
    if t_dim == 1:  # bias
        bias_o = weights_f.size()
        permute_weights = weights_f.reshape(1, -1)
    elif t_dim == 4:  # convolution or FC
        o, i, h_ks, w_ks = weights_f.size()
#         permute_weights = weights_f.reshape(16, -1)
        if channel:  # channel fiber
            permute_weights = weights_f.permute(0, 2, 3, 1)  # (o, i, h, w) ==> (o, h, w, i)
            o, h_ks, w_ks, i = permute_weights.size()
            permute_weights = weights_f.reshape(o* h_ks* w_ks, i)
        else:
            o, i, h_ks, w_ks = weights_f.size()
            permute_weights = weights_f.reshape(32, -1)  # (o, i, h, w) ==> (o, h, w, i)
            
            # (0, h, w, i) ==> (o, i, h, w)
#                 permute_weights = permute_weights.reshape(o * h_ks * w_ks, i)  # (o, h, w, I) ==> (o * h * w, i)
#             permute_weights = weights_f.reshape(o * i, h_ks * w_ks)  # (o, i, h, w) ==> (o * i, h * w)
    else:
        permute_weights = weights_f
        
    ternary_weights, cosine_similarity = TNT_convert(permute_weights)  # output the best ternary tensor and its cosine
                                                                       # similarity with floating type
    weights_t = scaling1(permute_weights, ternary_weights)

    if t_dim == 1:
        weights_t = weights_t.reshape(bias_o)
    elif t_dim == 4:
        if channel:
            weights_t = weights_t.reshape(o, h_ks, w_ks, i)
            weights_t = weights_t.permute(0, 3, 1, 2) # (0, h, w, i) ==> (o, i, h, w)
        else:
            weights_t = weights_t.reshape(o, i, h_ks, w_ks)
    return weights_t
