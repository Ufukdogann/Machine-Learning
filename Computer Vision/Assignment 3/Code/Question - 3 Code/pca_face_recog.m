% Load the dataset
% Train data
folder = 'faces/train/';
names = dir(folder);
train_data = zeros(16, 50, 64, 64);
for i=3:size(names, 1)
    samples = dir(fullfile(folder, names(i).name));
    for j=3:size(samples, 1)
        train_data(i-2, j-2, :, :) = rgb2gray(imread(fullfile(folder, names(i).name, samples(j).name)));
    end
end

% Test data
folder = 'faces/test/';
names = dir(folder);
test_data = zeros(16, 10, 64, 64);
for i=3:size(names, 1)
    samples = dir(fullfile(folder, names(i).name));
    for j=3:size(samples, 1)
        test_data(i-2, j-2, :, :) = rgb2gray(imread(fullfile(folder, names(i).name, samples(j).name)));
    end
 end
%% Prepare the data
train_data_sz = size(train_data);
X_train = reshape(train_data, [train_data_sz(1) * train_data_sz(2), train_data_sz(3) * train_data_sz(4)]);
Y_train = repmat(1:train_data_sz(1), [train_data_sz(2), 1]);
Y_train = Y_train(:);

test_data_sz = size(test_data);
X_test = reshape(test_data, [test_data_sz(1) * test_data_sz(2), test_data_sz(3) * test_data_sz(4)]);
Y_test = repmat(1:test_data_sz(1), [test_data_sz(2), 1]);
Y_test = Y_test(:);
%% Train PCA
[pca_W, pca_mean] = pca_fit(X_train, 'svd');
%% Visualize PCA components
figure(1);
for i=1:10
    subplot(1,10,i);
    img = reshape(pca_W(:, i), [64, 64]);
    imshow(img, [min(img(:)), max(img(:))]);
end
%% Plot captured variance
[evecs, evals] = pca_svd(X_train - mean(X_train, 1));
evals = evals' * evals;
evals = diag(evals);
d = 500;

ns=1:100:d;
capvar = zeros(length(ns), 1);

for i=1:length(ns)
    n = ns(i);
    capvar(i) = sum(evals(1:n)) / sum(evals);
end

figure(4);
plot(ns, capvar);

%% Reconstruct data with only a few princial components    
n_components = 500;
X_train_proj = pca_transform(X_train, n_components, pca_W, pca_mean);
X_train_rec = pca_reconstruct(X_train_proj, pca_W, pca_mean);

samples_per_class = 10;
figure(2);
for i=1:16
    idx = i;
    subplot(6, 16, idx);
    idx = (i-1) * samples_per_class + 1;
    img = reshape(X_train(idx, :), [64, 64]);
    imshow(img, [min(img(:)), max(img(:))]);
    
    idx = 16 + i;
    subplot(6, 16, idx);
    idx = (i-1) * samples_per_class + 1;
    img = reshape(X_train_rec(idx, :), [64, 64]);
    imshow(img, [min(img(:)), max(img(:))]);
    
    idx = 32 + i;
    subplot(6, 16, idx);
    idx = (i-1) * samples_per_class + 2;
    img = reshape(X_train(idx, :), [64, 64]);
    imshow(img, [min(img(:)), max(img(:))]);
    
    idx = 48 + i;
    subplot(6, 16, idx);
    idx = (i-1) * samples_per_class + 2;
    img = reshape(X_train_rec(idx, :), [64, 64]);
    imshow(img, [min(img(:)), max(img(:))]);
    
    idx = 64 + i;
    subplot(6, 16, idx);
    idx = (i-1) * samples_per_class + 3;
    img = reshape(X_train(idx, :), [64, 64]);
    imshow(img, [min(img(:)), max(img(:))]);
    
    idx = 80 + i;
    subplot(6, 16, idx);
    idx = (i-1) * samples_per_class + 3;
    img = reshape(X_train_rec(idx, :), [64, 64]);
    imshow(img, [min(img(:)), max(img(:))]);

end

%% Plot reconstruction error
[N, d] = size(X_train);
ns=1:100:d;
errors = zeros(length(ns), 1);
for i=1:length(ns)
    n = ns(i);
    X_train_proj = pca_transform(X_train, n, pca_W, pca_mean);
    X_train_rec = pca_reconstruct(X_train_proj, pca_W, pca_mean);
    
    errors(i) = mean((X_train_rec - X_train).^2, 'all');
end
figure(3);
plot(ns, errors);

%% Perform recognition based on eigenfaces
n_components = 50;
[pca_W, pca_mean] = pca_fit(X_train, 'svd');
X_train_proj = pca_transform(X_train, n_components, pca_W, pca_mean);
X_test_proj = pca_transform(X_test, n_components, pca_W, pca_mean);

dists = compute_distances(X_test_proj, X_train_proj);
Y_preds = predict_labels(dists, Y_train, 20);

num_test = length(Y_preds);
num_correct = sum(Y_preds == Y_test);
accuracy = num_correct / num_test;

fprintf('%d / %d correct (accuracy: %f)\n', num_correct, num_test, accuracy);

%% Bonus - perform recognition based on Fisherfaces (LDA)
% YOUR CODE GOES HERE