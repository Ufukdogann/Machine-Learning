function [X_proj] = pca_transform(X, n_components, W_pca, mean_pca)
%     Center and project X onto a lower dimensional space using W_pca.
%     Args:
%         X: Array of shape (N, D). Each row is an example with D features.
%         n_components: number of principal components.
%         W_pca: PCA projection matrix
%         mean_pca: Mean computed during PCA fit.
%     Returns:
%         X_proj: Array of shape (N, n_components).

% YOUR CODE GOES HERE

wanted_columns = W_pca(:,1:n_components);
transform = (X - mean_pca) * wanted_columns;
X_proj = transform;
        
end

