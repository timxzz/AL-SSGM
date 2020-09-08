import numpy as np
from torch.autograd import Variable
import torch
from torch.nn import functional as F

def acquisition_random(model, unlabelled, n_new=20):
    """
    Input: model, unlablled data, and the number of data to acquire

    Output:
        the acquired data point,
        the acquired data label,
        and the acquired data index from the Unlabelled set
    """

    all_idx = np.array([])
    all_u = np.array([])
    all_v = np.array([])
    for u, v, u_idx in unlabelled:
        all_idx = np.concatenate((all_idx, u_idx.numpy()), axis=None) if all_idx.size else u_idx.numpy()
        all_u = np.vstack([all_u, u.numpy()]) if all_u.size else u.numpy()
        # If y is not in onehot format
        if len(v.shape) == 1:
            all_v = np.concatenate((all_v, v.numpy()), axis=None) if all_v.size else v.numpy()
        else:
            all_v = np.vstack([all_v, v.numpy()]) if all_v.size else v.numpy()

    if len(all_idx) == 0:
        return None
    elif len(all_idx) < n_new:
        n_new = len(all_idx)

    indices = np.arange(len(all_idx))
    np.random.shuffle(indices)
    acquired_idx = indices[:n_new]

    return all_u[acquired_idx], all_v[acquired_idx], all_idx[acquired_idx]


def acquisition_entropy(model, unlabelled, use_cuda, n_new=20, use_M2=False):
    """
    Input: model, unlablled data, and the number of data to acquire

    Output:
        the acquired data point,
        the acquired data label,
        and the acquired data index from the Unlabelled set
    """

    model.eval()

    all_scores = np.array([])
    all_u = np.array([])
    all_v = np.array([])
    all_idx = np.array([])
    for u, v, u_idx in unlabelled:
        all_idx = np.concatenate((all_idx, u_idx.numpy()), axis=None) if all_idx.size else u_idx.numpy()
        all_u = np.vstack([all_u, u.numpy()]) if all_u.size else u.numpy()
        # If y is not in onehot format
        if len(v.shape) == 1:
            all_v = np.concatenate((all_v, v.numpy()), axis=None) if all_v.size else v.numpy()
        else:
            all_v = np.vstack([all_v, v.numpy()]) if all_v.size else v.numpy()

        # Wrap in variables
        u = Variable(u.float())

        if use_cuda:
            u = u.cuda()
        torch.manual_seed(0)  # <----------------------------------- !!!!!!!!!!!!!!!! Remember to disable this!!!

        if use_M2:
            logits_v = model.classify(u, logits=True)
        else:
            logits_v = model(u, logits=True)
        prob_v = F.softmax(logits_v, dim=-1)
        logprob_v = F.log_softmax(logits_v, dim=-1)

        H = -torch.sum(prob_v * logprob_v, dim=-1)

        all_scores = np.concatenate((all_scores, H.data.cpu().numpy()), axis=None) if all_scores.size else H.data.cpu().numpy()

    if len(all_idx) == 0:
        return None
    elif len(all_idx) < n_new:
        n_new = len(all_idx)

    # ha = np.argsort(-all_elbo)
    # print(np.sort(-all_elbo))
    # print(all_idx[ha])
    acquired_idx = np.argsort(-all_scores)[:n_new]  # Add negation for descending order

    return all_u[acquired_idx], all_v[acquired_idx], all_idx[acquired_idx]


def acquisition_gamma(model, unlabelled, use_cuda, gamma=1, n_new=20):
    """
    Input: model, unlablled data, and the number of data to acquire

    Output:
        the acquired data point,
        the acquired data label,
        and the acquired data index from the Unlabelled set
    """

    model.eval()

    all_elbo = np.array([])
    all_u = np.array([])
    all_v = np.array([])
    all_idx = np.array([])
    for u, v, u_idx in unlabelled:
        all_idx = np.concatenate((all_idx, u_idx.numpy()), axis=None) if all_idx.size else u_idx.numpy()
        all_u = np.vstack([all_u, u.numpy()]) if all_u.size else u.numpy()
        # If y is not in onehot format
        if len(v.shape) == 1:
            all_v = np.concatenate((all_v, v.numpy()), axis=None) if all_v.size else v.numpy()
        else:
            all_v = np.vstack([all_v, v.numpy()]) if all_v.size else v.numpy()
        # Wrap in variables
        u = Variable(u.float())

        if use_cuda:
            u = u.cuda()
        torch.manual_seed(0)  # <----------------------------------- !!!!!!!!!!!!!!!! Remember to disable this!!!
        elbo = model(u, gamma=gamma, in_batch=False)
        all_elbo = np.concatenate((all_elbo, elbo.data.cpu().numpy()), axis=None) if all_elbo.size else elbo.data.cpu().numpy()

    if len(all_idx) == 0:
        return None
    elif len(all_idx) < n_new:
        n_new = len(all_idx)

    # ha = np.argsort(-all_elbo)
    # print(np.sort(-all_elbo))
    # print(all_idx[ha])
    acquired_idx = np.argsort(-all_elbo)[:n_new]  # Add negation for descending order

    return all_u[acquired_idx], all_v[acquired_idx], all_idx[acquired_idx]


def acquisition_entropyXdensity(model, unlabelled, use_cuda, n_new=20):
    """
    Input: model, unlablled data, and the number of data to acquire

    Output:
        the acquired data point,
        the acquired data label,
        and the acquired data index from the Unlabelled set
    """

    model.eval()

    all_scores = np.array([])
    all_u = np.array([])
    all_v = np.array([])
    all_idx = np.array([])
    for u, v, u_idx in unlabelled:
        all_idx = np.concatenate((all_idx, u_idx.numpy()), axis=None) if all_idx.size else u_idx.numpy()
        all_u = np.vstack([all_u, u.numpy()]) if all_u.size else u.numpy()
        # If y is not in onehot format
        if len(v.shape) == 1:
            all_v = np.concatenate((all_v, v.numpy()), axis=None) if all_v.size else v.numpy()
        else:
            all_v = np.vstack([all_v, v.numpy()]) if all_v.size else v.numpy()
        # Wrap in variables
        u = Variable(u.float())

        if use_cuda:
            u = u.cuda()
        torch.manual_seed(0)  # <----------------------------------- !!!!!!!!!!!!!!!! Remember to disable this!!!
        entropy = model(u, gamma=0, in_batch=False)
        density = model(u, gamma=1, in_batch=False)
        scores = entropy * density
        all_scores = np.concatenate((all_scores, scores.data.cpu().numpy()), axis=None) if all_scores.size else scores.data.cpu().numpy()

    if len(all_idx) == 0:
        return None
    elif len(all_idx) < n_new:
        n_new = len(all_idx)

    # ha = np.argsort(-all_elbo)
    # print(np.sort(-all_elbo))
    # print(all_idx[ha])
    acquired_idx = np.argsort(-all_scores)[:n_new]  # Add negation for descending order

    return all_u[acquired_idx], all_v[acquired_idx], all_idx[acquired_idx]


def acquisition_ProbEntropyMixDensity(model, unlabelled, use_cuda, densityProb, n_new=20):
    """
    Input: model, unlablled data, and the number of data to acquire

    Output:
        the acquired data point,
        the acquired data label,
        and the acquired data index from the Unlabelled set
    """
    if np.random.uniform() <= densityProb:
        gamma = 1
    else:
        gamma = 0

    model.eval()

    all_scores = np.array([])
    all_u = np.array([])
    all_v = np.array([])
    all_idx = np.array([])
    for u, v, u_idx in unlabelled:
        all_idx = np.concatenate((all_idx, u_idx.numpy()), axis=None) if all_idx.size else u_idx.numpy()
        all_u = np.vstack([all_u, u.numpy()]) if all_u.size else u.numpy()
        # If y is not in onehot format
        if len(v.shape) == 1:
            all_v = np.concatenate((all_v, v.numpy()), axis=None) if all_v.size else v.numpy()
        else:
            all_v = np.vstack([all_v, v.numpy()]) if all_v.size else v.numpy()
        # Wrap in variables
        u = Variable(u.float())

        if use_cuda:
            u = u.cuda()
        torch.manual_seed(0)  # <----------------------------------- !!!!!!!!!!!!!!!! Remember to disable this!!!
        scores = model(u, gamma=gamma, in_batch=False)
        all_scores = np.concatenate((all_scores, scores.data.cpu().numpy()), axis=None) if all_scores.size else scores.data.cpu().numpy()

    if len(all_idx) == 0:
        return None
    elif len(all_idx) < n_new:
        n_new = len(all_idx)

    # ha = np.argsort(-all_elbo)
    # print(np.sort(-all_elbo))
    # print(all_idx[ha])
    acquired_idx = np.argsort(-all_scores)[:n_new]  # Add negation for descending order

    return all_u[acquired_idx], all_v[acquired_idx], all_idx[acquired_idx]


def acquisition_EntropyAndDensity(model, unlabelled, use_cuda, propo, entropy_inter_density=True, n_new=20):
    """
    Input: model, unlablled data, and the number of data to acquire

    Output:
        the acquired data point,
        the acquired data label,
        and the acquired data index from the Unlabelled set


    """

    model.eval()

    all_entropy = np.array([])
    all_density = np.array([])
    all_u = np.array([])
    all_v = np.array([])
    all_idx = np.array([])
    for u, v, u_idx in unlabelled:
        all_idx = np.concatenate((all_idx, u_idx.numpy()), axis=None) if all_idx.size else u_idx.numpy()
        all_u = np.vstack([all_u, u.numpy()]) if all_u.size else u.numpy()
        # If y is not in onehot format
        if len(v.shape) == 1:
            all_v = np.concatenate((all_v, v.numpy()), axis=None) if all_v.size else v.numpy()
        else:
            all_v = np.vstack([all_v, v.numpy()]) if all_v.size else v.numpy()
        # Wrap in variables
        u = Variable(u.float())

        if use_cuda:
            u = u.cuda()
        torch.manual_seed(0)  # <----------------------------------- !!!!!!!!!!!!!!!! Remember to disable this!!!
        entropy = model(u, gamma=0, in_batch=False)
        all_entropy = np.concatenate((all_entropy, entropy.data.cpu().numpy()),
                                     axis=None) if all_entropy.size else entropy.data.cpu().numpy()
        density = model(u, gamma=1, in_batch=False)
        all_density = np.concatenate((all_density, density.data.cpu().numpy()),
                                     axis=None) if all_density.size else density.data.cpu().numpy()

    if entropy_inter_density:
        all_score1 = all_entropy
        all_score2 = all_density
    else:
        all_score1 = all_density
        all_score2 = all_entropy

    if len(all_idx) == 0:
        return None
    elif len(all_idx) < n_new:
        n_new = len(all_idx)

    top_n_scores1 = int(len(all_idx) * propo)

    if top_n_scores1 == 0:
        acquired_idx = np.argsort(-all_score1)[:n_new]  # Add negation for descending order
        return all_u[acquired_idx], all_v[acquired_idx], all_idx[acquired_idx]
    elif top_n_scores1 < n_new:
        n_new = top_n_scores1

    acquired_idx_score1 = np.argsort(-all_score1)[:top_n_scores1]
    acquired_idx_score2 = np.argsort(-all_score2)

    # Calculate intersection
    bmask = np.full((len(all_idx),), False)
    bmask[acquired_idx_score1] = True
    acquired_idx_inter = acquired_idx_score2[bmask[acquired_idx_score2]]

    acquired_idx = acquired_idx_inter[:n_new]

    return all_u[acquired_idx], all_v[acquired_idx], all_idx[acquired_idx]