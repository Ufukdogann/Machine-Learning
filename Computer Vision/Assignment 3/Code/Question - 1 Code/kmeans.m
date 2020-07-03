function [memberships] = kmeans(features,k,num_iters)
% Use kmeans algorithm to group features into k clusters.
%    K-Means algorithm can be broken down into following steps:
%         1. Randomly initialize cluster centers
%         2. Assign each point to the closest center
%         3. Compute new center of each cluster
%         4. Stop if cluster assignments did not change
%         5. Go to step 2
%     Args:
%         features - Array of N features vectors. Each row represents a feature
%             vector.
%         k - Number of clusters to form.
%         num_iters - Maximum number of iterations the algorithm will run.
%     Returns:
%         memberships - Array representing cluster assignment of each point.
%             (e.g. i-th point is assigned to cluster memberships[i])
    
[N, D] = size(features);
assert(N >= k, 'Number of clusters cannot be greater than number of points');

% YOUR CODE GOES HERE


%initCentroids
centroids = [];

for i=1:k
    randidx = randperm(size(features,1));
    centroids = features(randidx(1:k), :);
end

for z = 1 : num_iters

%getClosestCentroids
indices = zeros(size(features,1), 1);

    for i=1:N
        a = 1;
        min_dist = sum((features(i,:) - centroids(1,:)) .^ 2);
        for j=2:k
            dist = sum((features(i,:) - centroids(j,:)) .^ 2);
            if(dist < min_dist)
            min_dist = dist;
            a = j;
            end
        end
        indices(i) = a;
    end
    
%computeCentroids
    for i=1:k
        xi = features(indices==i,:);
        ck = size(xi,1);
        if D==3
            centroids(i, :) = (1/ck) * [sum(xi(:,1)) sum(xi(:,2)) sum(xi(:,3))];
        elseif D==2
            centroids(i, :) = (1/ck) * [sum(xi(:,1)) sum(xi(:,2))];
        end
    end

end

memberships = indices;

end

