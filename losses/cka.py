import torch

def hsic0(gram_x: torch.Tensor, gram_y: torch.Tensor) -> torch.Tensor:
    """Compute the single sample version of the Hilbert-Schmidt Independence Criterion on Gram matrices.
    Args:
        gram_x: batch of Gram matrices of shape (n, n).
        gram_y: batch of Gram matrices of shape (n, n)
    """
    if len(gram_x.size()) != 2 or gram_x.size() != gram_y.size():
        raise ValueError("Invalid size for one of the two input tensors.")

    n = gram_x.shape[-1]

    # centering matrix
    H = torch.eye(n) - torch.ones_like(gram_x) / n

    kl = torch.matmul(torch.matmul(gram_x, H), torch.matmul(gram_y, H))
    trace_kl = kl.diagonal(dim1=0, dim2=1).sum()

    return trace_kl / ((n-1)**2)

def hsic1(gram_x: torch.Tensor, gram_y: torch.Tensor) -> torch.Tensor:
    """Compute the batched version of the Hilbert-Schmidt Independence Criterion on Gram matrices.

    This version is based on
    https://github.com/numpee/CKA.pytorch/blob/07874ec7e219ad29a29ee8d5ebdada0e1156cf9f/cka.py#L107.

    Args:
        gram_x: batch of Gram matrices of shape (bsz, n, n).
        gram_y: batch of Gram matrices of shape (bsz, n, n).

    Returns:
        a tensor with the unbiased Hilbert-Schmidt Independence Criterion values.

    Raises:
        ValueError: if ``gram_x`` and ``gram_y`` do not have the same shape or if they do not have exactly three
        dimensions.
    """
    if len(gram_x.size()) != 3 or gram_x.size() != gram_y.size():
        raise ValueError("Invalid size for one of the two input tensors.")

    n = gram_x.shape[-1]

    # Fill the diagonal of each matrix with 0
    gram_x.diagonal(dim1=-1, dim2=-2).fill_(0)
    gram_y.diagonal(dim1=-1, dim2=-2).fill_(0)

    # Compute the product between k (i.e.: gram_x) and l (i.e.: gram_y)
    kl = torch.bmm(gram_x, gram_y)

    # Compute the trace (sum of the elements on the diagonal) of the previous product, i.e.: the left term
    trace_kl = kl.diagonal(dim1=-1, dim2=-2).sum(-1).unsqueeze(-1).unsqueeze(-1)

    # Compute the middle term
    middle_term = gram_x.sum((-1, -2), keepdim=True) * gram_y.sum((-1, -2), keepdim=True)
    middle_term /= (n - 1) * (n - 2)

    # Compute the right term
    right_term = kl.sum((-1, -2), keepdim=True)
    right_term *= 2 / (n - 2)

    # Put all together to compute the main term
    main_term = trace_kl + middle_term - right_term

    # Compute the hsic values
    out = main_term / (n**2 - 3 * n)
    return out.squeeze(-1).squeeze(-1)

def cka_batch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the minibatch version of CKA from Nguyen et al. (https://arxiv.org/abs/2010.15327).

    This computation is performed with linear kernel and by calculating HSIC_1.

    Args:
        x: tensor of shape (bsz, n, j).
        y: tensor of shape (bsz, n, k).

    Returns:
        a float tensor in [0, 1] that is the CKA value between the two given tensors.
    """
    # Build the Gram matrices by applying the linear kernel
    gram_x = calculate_gram_batch(x) # XX^T (b, n, n)
    gram_y = calculate_gram_batch(y) # YY^T (b, n, n)

    # Compute the HSIC values for the entire batches
    hsic1_xy = hsic1(gram_x, gram_y)
    hsic1_xx = hsic1(gram_x, gram_x)
    hsic1_yy = hsic1(gram_y, gram_y)

    # Compute the CKA value
    cka = hsic1_xy.sum() / (hsic1_xx.sum() * hsic1_yy.sum()).sqrt()
    return cka

def cka_single(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the minibatch version of CKA from Nguyen et al. (https://arxiv.org/abs/2010.15327).

    This computation is performed with linear kernel and by calculating HSIC_1.

    Args:
        x: tensor of shape ( n, j).
        y: tensor of shape ( n, k).

    Returns:
        a float tensor in [0, 1] that is the CKA value between the two given tensors.
    """
    # Build the Gram matrices by applying the linear kernel
    gram_x = calculate_gram_single(x) # XX^T
    gram_y = calculate_gram_single(y) # YY^T

    # Compute the HSIC values for the entire batches
    hsic0_xy = hsic0(gram_x, gram_y)
    hsic0_xx = hsic0(gram_x, gram_x)
    hsic0_yy = hsic0(gram_y, gram_y)

    # Compute the CKA value
    cka = hsic0_xy / (hsic0_xx * hsic0_yy).sqrt()
    return cka

def calculate_gram_single(x: torch.Tensor):
    """
    Compute gram matrix XX^T
    :param x: tensor of shape (b, n, j)
    :return:
    """
    return torch.matmul(x, x.transpose(0, 1))

def calculate_gram_batch(x: torch.Tensor):
    """
    Compute gram matrix XX^T
    :param x: tensor of shape (b, n, j)
    :return:
    """
    return torch.bmm(x, x.transpose(1, 2))