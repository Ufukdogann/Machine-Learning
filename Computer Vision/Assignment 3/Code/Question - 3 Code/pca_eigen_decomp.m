function [evecs, evals] = pca_eigen_decomp(X)
%         Performs eigendecompostion of feature covariance matrix.
%         Args:
%             X: Zero-centered data array, each ROW containing a data point.
%                Array of shape (N, D).
%         Returns:
%             evecs: Eigenvectors of covariance matrix of X. Eigenvectors are
%                     sorted in descending order of corresponding eigenvalues. Each
%                     column contains an eigenvector. Array of shape (D, D).
%             evals: Eigenvalues of covariance matrix of X. Eigenvalues are
%                     sorted in descending order. Array of shape (D,).
        
    % YOUR CODE GOES HERE
    meanval = mean(X);
   
    d=X-meanval;
    %d=X-repmat(meanval,1,c);
    co = d'*d;
    %co=(1/c-1)*d*d';
    [eigvector,eigval]=eig(co);
    eigval = diag(eigval);
    
    [~,indx] = sort(eigval, 'descend');
    A = eigval;
    
    for i=1:indx(1)
        A(:,i) = eigvector(:,indx(i));
    end
    
    evecs = A;
    evals = eigval;
    
end

