function [X] = pca_reconstruct(X_proj, W_pca, mean_pca)
%     Do the exact opposite of method `transform`: try to reconstruct the original features.
%     Given the X_proj of shape (N, n_components) obtained from the output of `transform`,
%     we try to reconstruct the original X.
%     Args:
%         X_proj: Array of shape (N, n_components). Each row is an example with D features.
%         W_pca: PCA projection matrix
%         mean_pca: Mean computed during PCA fit.
%     Returns:
%         X: Array of shape (N, D).
        
    % YOUR CODE GOES HERE
    
    [rows, columns] = size(X_proj);
    wanted_columns = W_pca(:,1:columns);
    X = X_proj*wanted_columns';
    X = X + mean_pca;
    
    
end

