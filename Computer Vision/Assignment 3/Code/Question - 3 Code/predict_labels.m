function [y_pred] = predict_labels(dists, y_train, k)
    [num_test, num_train] = size(dists);
    
    y_pred = zeros(num_test, 1);
    for i=1:num_test
        [vals, idxs] = sort(dists(i, :));
        y_pred_tmp = y_train(idxs(1:k));
        
        [y_pred_occurs, y_pred_classes] = hist(y_pred_tmp, unique(y_pred_tmp));
        [maxval, maxidx] = max(y_pred_occurs);
        y_pred(i) = y_pred_classes(maxidx);
    end
end

