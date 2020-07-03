picture  = imread('example_2.jpeg');
gray_picture = double(picture);

[x_size, y_size] = size(gray_picture);
[Y,X] = meshgrid(1:x_size,1:x_size);

radius = x_size/2 - 10;
center = [x_size/2, x_size/2];
number_of_iterations = 300;
m = 2;
delta_t = 1;


phi_gray_picture = sqrt((X - center(1)).^2 + (Y - center(2)).^2) - radius;

convolutional_picture = imgaussfilt(gray_picture,5);

[gx,gy] = imgradientxy(convolutional_picture, 'central');
g = 1 ./ 1 + (gx.^2 + gy.^2);

matrix_size = x_size * y_size; 

A_row = zeros(matrix_size,3);
A_column = zeros(matrix_size,3);

new_phi = phi_gray_picture;
    
flatten_column_c = g(:);
A_column(:,1) = (flatten_column_c + circshift(flatten_column_c,-1))/2;
A_column(:,3) = (flatten_column_c + circshift(flatten_column_c,1))/2;
A_column(:,2) = - (A_column(:,1) + A_column(:,3));
    
c_transpose = transpose(g);
flatten_row_c = c_transpose(:);
A_row(:,1) = (flatten_row_c + circshift(flatten_row_c,-1))/2;
A_row(:,3) = (flatten_row_c + circshift(flatten_row_c,1))/2;
A_row(:,2) = - (A_row(:,1) + A_row(:,3));
    
A_row_new =  - (m^2 * delta_t * A_row);
A_row_new(:,2) = (A_row_new(:,2) + m);
    
A_column_new =  - (m^2 * delta_t * A_column);
A_column_new(:,2) = (A_column_new(:,2) + m);
    

for i = 1:number_of_iterations
    
    new_phi_transpose = transpose(new_phi);
    flatten_gray_picture_row = new_phi_transpose(:);
    flatten_gray_picture_column = new_phi(:);
    
    new_image_row = tridiag(A_row_new(:,2), A_row_new(:,3), A_row_new(:,1), flatten_gray_picture_row);
    new_image_column = tridiag(A_column_new(:,2), A_column_new(:,3), A_column_new(:,1), flatten_gray_picture_column);
    
    new_image_column = reshape(new_image_column,x_size,y_size);
    new_image_row = transpose(reshape(new_image_row,y_size,x_size));
    new_phi = new_image_row + new_image_column;
    
    m_phi = abs(new_phi) < 1;
    db = bwdist(m_phi);
    mask_inside = imfill(m_phi, 'holes');
    db(mask_inside) = -1*db(mask_inside);
    new_phi = db;
    
    imshow(uint8(new_phi ~= 0).*uint8(gray_picture) + uint8(new_phi == 0) * 255)
end


