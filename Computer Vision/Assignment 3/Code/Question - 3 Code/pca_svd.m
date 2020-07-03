function [vecs, vals] = pca_svd(X)
%     Performs Singular Value Decomposition (SVD) of X.
%     Args:
%         X: Zero-centered data array, each ROW containing a data point.
%            Array of shape (N, D).
%     Returns:
%         vecs: right singular vectors. Array of shape (D, D)
%         vals: singular values. Array of shape (K,) where K = min(N, D)
 
    % YOUR CODE GOES HERE
    meanval = mean(X);
    d=X-meanval;
    [u,eigvl,eigvector]=svd(d);
    vecs = eigvector;
    vals = eigvl;
    
    
end

