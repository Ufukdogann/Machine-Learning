picture  = imread('images/mri.png');
gray_picture = im2double(rgb2gray(picture));

K = 0.1;
number_of_iterations = 200;
delta_t = 5;
new_image = gray_picture;

for i = 1:number_of_iterations
    [gx,gy] = imgradientxy(new_image,'central');
    c = 1 ./  (1 + (((gx.^2 + gy.^2).^(1/2))./K).^2);
    [cx,cy] = imgradientxy(c,'central');
    [gxx,gyx] = imgradientxy(gx,'central');
    [gyx,gyy] = imgradientxy(gy,'central');
    divergence = ((cx.*gx) + (c.*gxx)) + ((cy.*gy) + (c.*gyy));
    new_image  = (divergence .* delta_t) + new_image;
end

imshow(new_image)