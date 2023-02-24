function [] = Pipeline()
%=========================================================================
% Name: Pipeline
%
% Description: Menu-driven image processing tools
%
% Licence: 
%
%    This program is free software: you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%    (at your option) any later version.
%
%    This program is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with this program.  If not, see <http://www.gnu.org/licenses/>
%=========================================================================
    
    global img_current img_master img_info;

    status = 1;
    
    %Enter main control loop.
    while status == 1
        
        % Generate menu
        choice = menu('Image Processing Tools','Exit', 'Load Image',...
            'Save Image', 'Gamma Correction','White Balance',...
            'Current Image', 'Original Image', 'Current & Original',...
            'Reset', 'Histograms (Current)', 'Histograms (Original)',...
            'Equalize Histograms','Contrast-Stretch',...
            'Edge-Enhance (Laplacian)', 'Band-Rejection Filter',...
            'High / Low Pass Filter', 'Fourier Transform (RGB)',...
            'HistoSquash (Adjust)', 'Dehaze');

        % Handle user interaction.
        [img_current, img_master, img_info status] = EvalChoice(choice,...
            img_current, img_master, img_info);
    
    end
       
end

%% Subfunction - LoadImage
function [img_out, img_info] = LoadImage()
   
    % Create file dialogue.
    [file, path] = uigetfile( ...
        {'*.jpg', '*.jpeg)'; ...
        '*.*','All Files (*.*)'}, ...
        'Select Image');

    % If "Cancel" is pressed then return.
    if isequal([file,path],[0,0])
    
        return
    
    % Else load the image.
    else
    
        % Concatenate 'path' and 'file'.
        img_path = fullfile(path, file);

        img_out = imread(img_path);
        img_out = im2double(img_out);

        % Get image meta-data.
        img_info = imfinfo(img_path);
    
    end
    
end

%% Subfunction - EvalChoice
function [img_current, img_master, img_info, status] = EvalChoice(choice,...
    img_current, img_master, img_info)

    status = 1;

    % Exit program.
    if choice == 0
        close all;
        status = -1;
    end

    if choice == 1
        close all;
        status = -1;
    end

    % Load image.
    if choice == 2
        [img_master, img_info] = LoadImage();
    
        % Store a master copy of the image.
        img_current = img_master;
    end
    
    % Save processed image.
    if choice == 3 
        close all;
        SaveImage(img_current); 
    end
    
    % Perform gamma correction
    if choice == 4
        close all;
        img_current = GammaCorrectionAll(img_current, img_info);
    end

    % Perform white balance
    if choice == 5
        close all;
        img_current = WhiteBalance(img_current);
    end

    % Show current image.
    if choice == 6
        figure;
        imshow(img_current);
        title('Current Image');
    end
    
    % Show original image.
    if choice == 7
        figure;
        imshow(img_master);
        title('Original Image');
    end
    
    % Compare original and processed images.
    if choice == 8
        close all;
        CompareImages(img_master, 'Original Image', img_current,...
            'Processed Image');
    end

    % Reset current image back to the original.
    if choice == 9
        close all;
        img_current = img_master;
    end
    
    % Display histograms for current image.
    if choice == 10
        Histograms(img_current);
    end
    
    % Display histograms for original image.
    if choice == 11
        Histograms(img_master);
    end
    
    % Equalize Histograms.
    if choice == 12
        img_current(:,:,1) = histeq(img_current(:,:,1));
        img_current(:,:,2) = histeq(img_current(:,:,2));
        img_current(:,:,3) = histeq(img_current(:,:,3));
    end
    
    % Contrast Stretch.
    if choice == 13
       img_current = imadjust(img_current, stretchlim(img_current), []);
    end
    
    % Laplacian Edge-Enhancement.
    if choice == 14
       img_current = LaplacianEdgeEnhancement(img_current); 
    end
    
    % BandReject filter (fft2)
    if choice == 15
       % Prompt user for inner and outer radius of band/annulus.
       r_inner = input('Inner radius?: ');
       r_outer = input('Outer radius?: ');
       img_current(:,:,1) = BandReject(img_current(:,:,1), img_info,...
          r_inner, r_outer);
       img_current(:,:,2) = BandReject(img_current(:,:,2), img_info,...
          r_inner, r_outer);
       img_current(:,:,3) = BandReject(img_current(:,:,3), img_info,...
          r_inner, r_outer);
    end
    
    % FlexiFilter
    if choice == 16
        % Prompt user for parameters and filter type.
        filter_type = input('Low pass (lp), high pass (hp)?: ');
        filter_func = input('gaussian, ideal, butterworth (btw)?: ');
        filter_width = input('Width?: ');
        
        img_current(:,:,1) = FlexiFilter(img_current(:,:,1),img_info,...
            filter_type,filter_func,filter_width);
        img_current(:,:,2) = FlexiFilter(img_current(:,:,2),img_info,...
            filter_type,filter_func,filter_width);
        img_current(:,:,3) = FlexiFilter(img_current(:,:,3),img_info,...
            filter_type,filter_func,filter_width);
    end
    
    % FT
    if choice == 17
        ShowFourier(img_current);
    end
    
    % HistoSquash / Adjust
    if choice == 18
       img_current = HistoSquash(img_current);
    end
    
    % Dehaze
    if choice == 19
       img_current = Dehaze(img_current);
    end
    
end

%% Subfunction - SaveImage
function [] = SaveImage(img_current)
    
    % Create file dialogue.
    [file, path] = uiputfile( ...
        {'*.jpg', '*.jpeg)'; ...
        '*.*','All Files (*.*)'}, ...
        'Save Image');

    % If "Cancel" is pressed then return.
    if isequal([file,path],[0,0])
    
        return
    
    % Else save the image.
    else
    
        % Concatenate 'path' and 'file'.
        save_path = fullfile(path, file);

        % Export image
        imwrite(img_current, save_path);
    
    end
    
end

%% Subfunction - GammaCorrectionAll
function [img_out] = GammaCorrectionAll(img_in, img_info)

    % Specify values for RGB gamma correction
    % via sliders.
    corrector = GammaCorrector(img_in);
    
    % Wait until user closes window.
    waitfor(corrector.Figure);
    
    % Perform gamma correction on each colour channel.
    gamma = corrector.Correction
    
    for i = 1:3
       img_out(:,:,i) = GammaCorrection(img_in, img_info, i, gamma(i));
    end
    
end

%% Subfunction - GammaCorrection
function [channel_out] = GammaCorrection(img_in, img_info, channel, gamma)

    % Get image dimensions.
    height = img_info.Height;
    width = img_info.Width;

    % Pre-allocate array for better performance.
    channel_out = zeros(height, width);

    % Perform gamma correction to specified channel.
    channel_out(:,:) = imadjust(img_in(:,:,channel), [], [], gamma);
    
end

%% Subfunction - WhiteBalance
function [img_out] = WhiteBalance(img_in)

    % Get the RGB values of a user-selected white pixel
    f = figure
    white_rgb = impixel(img_in);
    close(f);
    
    % Normalize the RGB channels
    for i = 1:3
        img_out(:,:,i) = img_in(:,:,i) ./ white_rgb(i);
    end
   
end

%% Subfunction - CompareImages
function [] = CompareImages(img1, title1, img2, title2)

    figure;

    subplot(1,2,1);
    imshow(img1);
    title(title1);
    
    subplot(1,2,2);
    imshow(img2);
    title(title2)
    
end

%% Subfuntion - Histograms
function [] = Histograms(img_in)
% Produces a 2x2 grid of histograms, for the individual
% RGB colour channels and the weighted luminance.
% Luminance = 0.3 R + 0.59 G + 0.11 B

    % Make RGB histograms.
    figure;
    subplot(2,2,1);
    imhist(img_in(:,:,1));
    grid on
    xlabel('Red Intensity');
    ylabel('Counts');
    title('Red Intensity Histogram');

    subplot(2,2,2);
    imhist(img_in(:,:,2));
    grid on
    xlabel('Green Intensity');
    ylabel('Counts');
    title('Green Intensity Histogram');

    subplot(2,2,3);
    imhist(img_in(:,:,3));
    grid on
    xlabel('Blue Intensity');
    ylabel('Counts');
    title('Blue Intensity Histogram');

    % Make Luminance histogram.
    lum = (0.3*img_in(:,:,1))+(0.59*img_in(:,:,2))+(0.11*img_in(:,:,3));
    subplot(2,2,4);
    imhist(lum);
    grid on
    xlabel('Luminance');
    ylabel('Counts');
    title('Luminance');

end

%% Subfunction LaplacianEdgeEnhancement
function [img_out] = LaplacianEdgeEnhancement(img_in)
% The Laplacian of an image highlights
% regions of rapid intensity change and
% is therefore often used for edge detection

    % Generate Laplacian filter with centre -8
    % to produce sharper edges than the default -4.
    f = [1 1 1; 1 -8 1; 1 1 1];
    
    % Apply filter to each channel.
    img_out(:,:,1) = img_in(:,:,1) - ...
        imfilter(img_in(:,:,1), f, 'replicate');
        
    img_out(:,:,2) = img_in(:,:,2) - ...
        imfilter(img_in(:,:,2), f, 'replicate');
    
    img_out(:,:,3) = img_in(:,:,3) - ...
        imfilter(img_in(:,:,3), f, 'replicate');
    
end

%% Subfunction - BandReject
function [img_out] = BandReject(img_in, img_info, r_inner, r_outer)
% Removes spatial frequencies within an annulus

    master = img_in;
    master_info = img_info;
   
    % Compute Fourier transform
    f = im2double(master);
    F = fft2(f(:,:,1));
    Fc = fftshift(F);
    S = abs(Fc);
    S2 = log(1+S);
    
    % Create circular mask (disk)
    height = master_info.Height;
    width = master_info.Width;
    mask = ones(height, width);

    for y = 1:height
        x_low = (width/2) - sqrt(r_outer^2-(y-(height/2))^2);
        x_high = (width/2) + sqrt(r_outer^2-(y-(height/2))^2);
        mask(y, real(ceil(x_low)):real(floor(x_high))) = 0;

        x_low = (width/2) - sqrt(r_inner^2-(y-(height/2))^2);
        x_high = (width/2) + sqrt(r_inner^2-(y-(height/2))^2);
        mask(y, real(ceil(x_low)):real(floor(x_high))) = 1;
    end   
    
    % Apply mask
    S3 = S2.*mask;

    % Compute inverse Fourier transform
    S4 = ifft2(ifftshift(mask.*Fc));

    img_out = S4;
end

%% Subfunction FlexiFilter
function [img_out] = FlexiFilter(img_in,img_info,filter_type,filter_func,...
    filter_width)

    master = img_in;
    master_info = img_info;

    % Compute fft2 transform
    f = im2double(master);
    F = fft2(f(:,:,1));
    Fc = fftshift(F);
    S = abs(Fc);
    S2 = log(1+S);

    % Create circular mask (disk)
    height = master_info.Height;
    width = master_info.Width;

    if filter_type=='hp'
        mask = hpfilter(filter_func,height,width,filter_width);
    end

    if filter_type=='lp'
        mask = lpfilter(filter_func,height,width,filter_width);
    end

    mask = fftshift(mask);

    % Apply mask
    S3 = S2.*mask;

    % Compute inverse Fourier transform
    S4 = ifft2(ifftshift(mask.*Fc));

    img_out = S4;

end

%% Subfunction ShowFourier
function [] = ShowFourier(img_in)
  
    % Compute fft2 transform for each channel
    for n = 1:3
        F = fft2(img_in(1:200,1:200,n));
        Fc = fftshift(F);
        S = abs(Fc);
        S2 = log(1+S);
        figure;
        imshow(S2);
        title(['Fourier Plane (Complement) ', n]);
    end
    
end

%% Subfunction HistoSquash
function [img_out] = HistoSquash(img_in)
%Adjust histogram by mapping to domain [0.2,0.8]
    % Ask user for domain limits
    lower=input('Domain - lower limit?: ');
    upper=input('Domain - upper limit?: ');
    img_out=imadjust(img_in, [0 1], [lower upper]);
end

%% Dehaze
function [img_out] = Dehaze(img_in)

    % Convert to hsv
    hsv = rgb2hsv(img_in);

    %get the illuminance channel
    v = hsv(:,:,3);

    %find the max pixel value
    max_v = max(max(v));

    % find the position of pixels having this value.
    [r, c] = find(v == max_v);
    
    norm = img_in(r(1),c(1),1) + img_in(r(1),c(1),2) + img_in(r(1),c(1),3);
    
    alpha_r = img_in(r(1),c(1),1) / norm;
    alpha_g = img_in(r(1),c(1),2) / norm;  
    alpha_b = img_in(r(1),c(1),3) / norm;
        
    img_white(:,:,1) = img_in(:,:,1) ./ alpha_r;
    img_white(:,:,2) = img_in(:,:,3) ./ alpha_g;
    img_white(:,:,3) = img_in(:,:,2) ./ alpha_b;
        
    %%%%%%%%%%%%%%%55
    
    [x_max, y_max] = size(img_white);
    
    img_padded = padarray(img_white, [1 1]);
    for i = 2:x_max;
        for j = 2:y_max
            D_gamma(i, j) = 1;
        end
    end
    
    img_out = img_white;
end