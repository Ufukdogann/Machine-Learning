image = imread('image.jpg');

for z = 1:200
gray_image = rgb2gray(image);
gray_image = im2double(gray_image);

[gx, gy] = imgradientxy(gray_image);
energy_map = abs(gx) + abs(gy);
imshow(energy_map);

[rows, columns] = size(energy_map);

%minimum seam

[row_number, column_number] = size(energy_map);

M = energy_map;
backtrack = zeros(row_number, column_number);

for i = 2:row_number
    for j = 1:column_number
        if j==1
            [min_value,idx] = min(M(i-1,j:j+1));
            backtrack(i,j) = idx+j;
            min_energy = M(i-1, idx+j);
            
        elseif j==column_number
            [min_value,idx] = min(M(i - 1, j - 1:j));
            backtrack(i, j) = idx + j - 2;
            min_energy = M(i - 1, idx + j - 2);
            
        else 
            [min_value,idx] = min(M(i - 1, j - 1:j + 1));
            backtrack(i, j) = idx + j - 2;
            min_energy = M(i - 1, idx + j - 2);
        end
        
        M(i, j) = M(i,j) + min_energy;
        
    end

end

%crop columns

scale_c = 0.5;

    mask = ones(row_number,column_number);
    [rows,columns] = size(M);
    [min_value, j] = min(M(rows,:));

    for s = row_number:-1:1
        mask(s,j) = 0;
        j = backtrack(s,j);
    end
    [r,verticalSeam] = find(mask == 0);
    for c=1:length(verticalSeam)
        image(c,verticalSeam(c):end-1,:)=image(c,verticalSeam(c)+1:end,:);
    end
    imshow(mask);
    image = image(:,1:end-1,:); 
end

imshow(image)


