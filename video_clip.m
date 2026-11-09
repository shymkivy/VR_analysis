f_path = 'C:\Users\ys2605\Desktop\stuff\pres\11_9_26\vid\';
fname = "Desktop 2026.11.08 - 23.53.09.05.mp4";


[~, name, ~] = fileparts(fname);

vidObj = VideoReader(f_path + fname);


numFrames = vidObj.NumFrames;

height = vidObj.Height;
width = vidObj.Width;
fr = vidObj.FrameRate;

time_range = [66, 76];
time_range_fr = time_range*fr;

frame = read(vidObj, 1);
figure();
imshow(frame);

sy = [419,1134];
s1x = [679, 1958];
s2x = [650, 1601];
s3x = [1602, 2548];

if 0
    frame = read(vidObj, 1);
    figure();
    imshow(frame(sy(1):sy(2), s1x(1):s1x(2),:))
    
    frame = read(vidObj, 1);
    figure();
    imshow(frame(sy(1):sy(2), s2x(1):s2x(2),:))
    
    frame = read(vidObj, 1);
    figure();
    imshow(frame(sy(1):sy(2), s3x(1):s3x(2),:))
end

vid1_out = zeros(diff(sy)+1, diff(s1x)+1, 3, diff(time_range_fr), 'uint8');
vid2_out = zeros(diff(sy)+1, diff(s2x)+1, 3, diff(time_range_fr), 'uint8');
vid3_out = zeros(diff(sy)+1, diff(s3x)+1, 3, diff(time_range_fr), 'uint8');
for ii = 1:diff(time_range_fr)
    n_fr = ii + time_range_fr(1);
    frame = read(vidObj, n_fr);
    vid1_out(:,:,:,ii) = frame(sy(1):sy(2), s1x(1):s1x(2),:);
    vid2_out(:,:,:,ii) = frame(sy(1):sy(2), s2x(1):s2x(2),:);
    vid3_out(:,:,:,ii) = frame(sy(1):sy(2), s3x(1):s3x(2),:);
end

warning('off', 'all');
vid_out = VideoWriter(f_path + name + "_clip13", "MPEG-4"); % MPEG-4, Motion JPEG AVI, Archival, Uncompressed AVI
vid_out.FrameRate = vidObj.FrameRate;
open(vid_out);
writeVideo(vid_out,vid1_out);
close(vid_out);

vid_out = VideoWriter(f_path + name + "_clip2", "MPEG-4"); % MPEG-4, Motion JPEG AVI, Archival, Uncompressed AVI
vid_out.FrameRate = vidObj.FrameRate;
open(vid_out);
writeVideo(vid_out,vid2_out);
close(vid_out);

vid_out = VideoWriter(f_path + name + "_clip3", "MPEG-4"); % MPEG-4, Motion JPEG AVI, Archival, Uncompressed AVI
vid_out.FrameRate = vidObj.FrameRate;
open(vid_out);
writeVideo(vid_out,vid3_out);
close(vid_out);






