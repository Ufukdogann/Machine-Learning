picture  = imread('images/mri.png');
gray_picture = im2double(rgb2gray(picture));

K = 0.1;
m = 2;
delta_t = 5;

[x_size, y_size] = size(gray_picture);

number_of_iterations = 200;
matrix_size = x_size * y_size; 

A_row = zeros(matrix_size,3);
A_column = zeros(matrix_size,3);

transpose_gray_picture = transpose(gray_picture);
flatten_for_column_A = gray_picture(:);
flatten_for_row_A = transpose_gray_picture(:);

length_of_flatten_column_A = length(flatten_for_column_A);
new_image = gray_picture;

for i = 1:number_of_iterations
    
    [gx,gy] = imgradientxy(new_image,'central');
    c = 1 ./  (1 + (((gx.^2 + gy.^2).^(1/2))./K).^2);
    
    flatten_column_c = c(:);
    A_column(:,1) = (flatten_column_c + circshift(flatten_column_c,-1))/2;
    A_column(:,3) = (flatten_column_c + circshift(flatten_column_c,1))/2;
    A_column(:,2) = - (A_column(:,1) + A_column(:,3));
    
    c_transpose = transpose(c);
    flatten_row_c = c_transpose(:);
    A_row(:,1) = (flatten_row_c + circshift(flatten_row_c,-1))/2;
    A_row(:,3) = (flatten_row_c + circshift(flatten_row_c,1))/2;
    A_row(:,2) = - (A_row(:,1) + A_row(:,3));
    
    new_image_transpose = transpose(new_image);
    flatten_gray_picture_row = new_image_transpose(:);
    flatten_gray_picture_column = new_image(:);
    
    A_row_new =  - (m^2 * delta_t * A_row);
    A_row_new(:,2) = (A_row_new(:,2) + m);
    
    A_column_new =  - (m^2 * delta_t * A_column);
    A_column_new(:,2) = (A_column_new(:,2) + m);
    
    new_image_row = tridiag(A_row_new(:,2), A_row_new(:,3), A_row_new(:,1), flatten_gray_picture_row);
    new_image_column = tridiag(A_column_new(:,2), A_column_new(:,3), A_column_new(:,1), flatten_gray_picture_column);
    
    new_image_column = reshape(new_image_column,x_size,y_size);
    new_image_row = transpose(reshape(new_image_row,y_size,x_size));
    new_image = new_image_row + new_image_column;
    
end

imshow(new_image)