function [dists] = compute_distances(X1, X2)
    M = size(X1, 1);
    N = size(X2, 1);
        
    X1_square = sum(X1.^2, 2);
    X2_square = sum(X2.^2, 2);
    dists = X1_square - 2 * X1 * X2' + X2_square';
end

