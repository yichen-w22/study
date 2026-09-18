import numpy as np

def softmax_classifier(W, X, y, _lambda):
    """
      Softmax Classifier

      Inputs have dimension D, there are C classes, a minibatch have N examples.
      (In this homework, D = 784, C = 10)

      Inputs:
      - W: A numpy array of shape (D, C) containing weights.
      - input: A numpy array of shape (N, D) containing a minibatch of data.
      - label: A numpy array of shape (N, C) containing labels, label[i] is a
        one-hot vector, label[i][j]=1 means i-th example belong to j-th class.
      - _lambda: regularization strength, which is a hyerparameter.

      Returns:
      - loss: a single float number represents the average loss over the minibatch.
      - gradient: shape (D, C), represents the gradient with respect to weights W.
      - prediction: shape (N,), prediction[i]=c means i-th example belong to c-th class.
    """

    ############################################################################
    """
    W: (D, K)    权重（若已把 bias 拼进特征，W 的最后一行/列通常对应偏置）
    X: (N, D) 或 (D, N)
    y: 形状可以是
        - (N,) 的整型类别索引 [0..K-1]
        - (N, K) 的 one-hot
        - (K, N) 的 one-hot（会自动转置）
    _lambda: L2 正则系数（对应 0.5*λ||W||^2）

    返回:
      loss: 标量
      dW:   与 W 同形状
      pred: (N,) 预测类别索引
    """

    # ---- 1) 统一 X 形状为 (N, D) ----
    if X.ndim == 1:
        X = X[None, :]
    if X.shape[1] != W.shape[0]:
        # 说明传入是 (D, N)，转成 (N, D)
        X = X.T
    N, D = X.shape
    K = W.shape[1]

    # ---- 2) 统一 y 为 one-hot，形状 (N, K) ----
    if y.ndim == 1:          # 类别索引
        Y = np.zeros((N, K), dtype=float)
        Y[np.arange(N), y.astype(int)] = 1.0
    else:                    # 已是 one-hot
        Y = y
        if Y.shape == (K, N):
            Y = Y.T
        if Y.shape != (N, K):
            raise ValueError(f"Label shape must be (N,) or (N,K)/(K,N), got {y.shape}")

    # ---- 3) 前向：稳定 softmax ----
    scores = X @ W                          # (N, K)
    scores -= scores.max(axis=1, keepdims=True)  # 数值稳定
    exp_scores = np.exp(scores)
    P = exp_scores / exp_scores.sum(axis=1, keepdims=True)  # (N, K)

    # ---- 4) 交叉熵损失（取平均）+ L2 正则 (0.5*λ||W||^2) ----
    eps = 1e-12
    ce = -np.sum(Y * np.log(P + eps)) / N

    # 如果最后一列特征是常数 1（偏置），通常不对偏置做正则：自动检测
    reg_mask = np.ones_like(W)
    if np.allclose(X[:, -1], 1.0):
        reg_mask[-1, :] = 0.0

    reg = 0.5 * _lambda * np.sum((W * reg_mask) ** 2)
    loss = ce + reg

    # ---- 5) 反向：梯度 (向量化) ----
    dScores = (P - Y) / N                   # (N, K)
    dW = X.T @ dScores                      # (D, K)
    dW += _lambda * (W * reg_mask)

    # ---- 6) 预测标签 ----
    pred = np.argmax(P, axis=1)             # (N,)

   

    

    ############################################################################

    return loss, dW, pred
