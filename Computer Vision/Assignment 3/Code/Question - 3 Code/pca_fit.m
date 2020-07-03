function [W_pca,meanval] = pca_fit(X, method)
%     Fit the training data X using the chosen method.
%     Args:
%         X: Array of shape (N, D). Each of the N rows represent a data point.
%            Each data point contains D features.
%         method: Method to solve PCA. Must be one of 'svd' or 'eigen'.
%     Returns:
%         W_pca: projection matrix
%         meanval: mean of the data
        
    % YOUR CODE GOES HERE
    
    
    boolean = strcmp('svd', method);
    meanval = mean(X);
    
    if boolean == 0
        [evecs, evals] = pca_eigen_decomp(X);
        W_pca = evecs;
    else 
        [vecs,vals] = pca_svd(X);
        W_pca = vecs;
    end
    
    
end

