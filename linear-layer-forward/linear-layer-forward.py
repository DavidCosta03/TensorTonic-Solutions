def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    n, d_in = len(X), len(X[0])
    d_out = len(W[0])
    out = []
    for i in range(n):
        row = []
        for j in range(d_out):
            total = 0
            for k in range(d_in): 
                total += X[i][k] * W[k][j]
            total += b[j]
            row.append(total)
        out.append(row)
    return out