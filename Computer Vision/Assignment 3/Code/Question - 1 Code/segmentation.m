% Prepare clustering data
% Cluster 1
mean1 = [-1, 0];
cov1 = [[0.1, 0]; [0, 0.1]];
X1 = mvnrnd(mean1, cov1, 100);

% Cluster 2
mean2 = [0, 1];
cov2 = [[0.1, 0]; [0, 0.1]];
X2 = mvnrnd(mean2, cov2, 100);

% Cluster 3
mean3 = [1, 0];
cov3 = [[0.1, 0]; [0, 0.1]];
X3 = mvnrnd(mean3, cov3, 100);

% Cluster 4
mean4 = [0, -1];
cov4 = [[0.1, 0]; [0, 0.1]];
X4 = mvnrnd(mean4, cov4, 100);

% Merge two sets of data points
X = [X1; X2; X3; X4];

scatter(X(:, 1), X(:, 2));


% Run k-means
assignments = kmeans(X, 4, 100);

% Show results
hold on
scatter(X(assignments == 1, 1), X(assignments == 1, 2));
scatter(X(assignments == 2, 1), X(assignments == 2, 2));
scatter(X(assignments == 3, 1), X(assignments == 3, 2));
scatter(X(assignments == 4, 1), X(assignments == 4, 2));
hold off

%%
% Load the image
img = im2double(imread('train.jpg'));
img = imresize(img, 0.5);
imshow(img)


% Compute color features
color_features = reshape(img, size(img, 1) * size(img, 2), size(img, 3));

% Run the segmentation
assignments = kmeans(color_features, 16, 100);

% Show results
assignments = reshape(assignments, size(img, 1), size(img, 2));
imshow(assignments, [min(min(assignments)), max(max(assignments))]);
colormap('jet');


