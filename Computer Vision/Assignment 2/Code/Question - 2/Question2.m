picture = imread('image_mask.png');

redChannel = picture(:, :, 1);
greenChannel = picture(:, :, 2);
blueChannel = picture(:, :, 3);
mask = redChannel == 252 & greenChannel == 2 & blueChannel == 4;

gray_picture = rgb2gray(picture);
gray_picture = im2double(gray_picture);
gray_picture(mask) = 0;

n_iterations = 50;
new_image = gray_picture;
epsilon = 0.5;
tau = 0.01;

for i = 1:n_iterations
    [fx,fy] = imgradientxy(new_image,'central');
    [fxx,fxy] = imgradientxy(fx,'central');
    [fyx,fyy] = imgradientxy(fy,'central');
    new_image = new_image + tau * 2 * ((fxx.*(fy.^2) - (fyx+fxy).*fx.*fy + fyy.*(fx.^2) + (fxx + fyy)*epsilon)./((fx.^2 + fy.^2 + epsilon).^(3/2)));
    new_image = (new_image.*mask) + (gray_picture.*(~mask));
end
imshow(new_image)
