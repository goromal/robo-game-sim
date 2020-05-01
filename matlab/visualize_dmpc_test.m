logfile = '../py/dmpc_game.log';
make_video = true; % will run slower than real-time if true
videofile = 'minimal_game_2v2_100s.avi';
% These parameters should match what you ran your single-run sim with:
T = 100.0;
dt = 0.05;
max_score = 4;
bx0 = 0.0;
by0 = 0.0;

%%

close all
addpath(genpath('matlab_utilities'))

% Read output log file, sample at 20 FPS
logdata = read_log(logfile, 13);
t_raw   = logdata(1,:);
score_A = logdata(2,:);
score_B = logdata(3,:);
px_ball = logdata(4,:);
py_ball = logdata(5,:);
px_A1   = logdata(6,:);
py_A1   = logdata(7,:);
px_A2   = logdata(8,:);
py_A2   = logdata(9,:);
px_B1   = logdata(10,:);
py_B1   = logdata(11,:);
px_B2   = logdata(12,:);
py_B2   = logdata(13,:);
t = 0.0:0.05:t_raw(end);
score_A = interp1(t_raw, score_A, t);
score_B = interp1(t_raw, score_B, t);
px_ball = interp1(t_raw, px_ball, t);
py_ball = interp1(t_raw, py_ball, t);
px_A1   = interp1(t_raw, px_A1, t);
py_A1   = interp1(t_raw, py_A1, t);
px_A2   = interp1(t_raw, px_A2, t);
py_A2   = interp1(t_raw, py_A2, t);
px_B1   = interp1(t_raw, px_B1, t);
py_B1   = interp1(t_raw, py_B1, t);
px_B2   = interp1(t_raw, px_B2, t);
py_B2   = interp1(t_raw, py_B2, t);

% Animate
width = 10.0;
height = 5.0;
Prad = 0.200;
prad = 0.175;
figure('position',[10 10 2000 1000])
patch([-1 1 1 -1]*width/2, [-1 -1 1 1]*height/2, 'green');
hold on
patch(-width/2 + [-1 1 1 -1]*0.075, [-1 -1 1 1]*0.5, 'red');
patch( width/2 + [-1 1 1 -1]*0.075, [-1 -1 1 1]*0.5, 'blue');
tht = 0:0.01:2*pi;
circle_x = cos(tht);
circle_y = sin(tht);
plot(circle_x,circle_y,'w-',[0 0],[-1 1]*height/2,'w-','Linewidth',2)
textA = text(-0.48*width,0.47*height,'');
textB = text( 0.42*width,0.47*height,'');
blp = patch(bx0 + prad*circle_x,      by0 + prad*circle_y, 'black');
blc =  plot(bx0 + prad*0.85*circle_x, by0 + prad*0.85*circle_y, 'w-');
A1p = patch(-width/4 + Prad*circle_x,   height/4 + Prad*circle_y,   'red');
A1c =  plot(-width/4 + prad/2*circle_x, height/4 + prad/2*circle_y, 'k-');
A2p = patch(-width/4 + Prad*circle_x,   -height/4 + Prad*circle_y,   'red');
A2c =  plot(-width/4 + prad/2*circle_x, -height/4 + prad/2*circle_y, 'k-');
B1p = patch(width/4 + Prad*circle_x,   height/4 + Prad*circle_y,   'blue');
B1c =  plot(width/4 + prad/2*circle_x, height/4 + prad/2*circle_y, 'k-');
B2p = patch(width/4 + Prad*circle_x,   -height/4 + Prad*circle_y,   'blue');
B2c =  plot(width/4 + prad/2*circle_x, -height/4 + prad/2*circle_y, 'k-');
pbaspect([2 1 1])
xlim([-width/2 width/2])
ylim([-height/2 height/2])
set(gca,'xtick',[])
set(gca,'xticklabel',[])
set(gca,'ytick',[])
set(gca,'yticklabel',[])
hold off
for i = 1:1:length(t)
    title(sprintf('t = %f s',t(i)))
    set(blp, 'XData', px_ball(i) + prad*circle_x,      'YData', py_ball(i) + prad*circle_y)
    set(blc, 'XData', px_ball(i) + prad*0.85*circle_x, 'YData', py_ball(i) + prad*0.85*circle_y)
    set(A1p, 'XData', px_A1(i) + Prad*circle_x,   'YData', py_A1(i) + Prad*circle_y)
    set(A1c, 'XData', px_A1(i) + prad/2*circle_x, 'YData', py_A1(i) + prad/2*circle_y)
    set(A2p, 'XData', px_A2(i) + Prad*circle_x,   'YData', py_A2(i) + Prad*circle_y)
    set(A2c, 'XData', px_A2(i) + prad/2*circle_x, 'YData', py_A2(i) + prad/2*circle_y)
    set(B1p, 'XData', px_B1(i) + Prad*circle_x,   'YData', py_B1(i) + Prad*circle_y)
    set(B1c, 'XData', px_B1(i) + prad/2*circle_x, 'YData', py_B1(i) + prad/2*circle_y)
    set(B2p, 'XData', px_B2(i) + Prad*circle_x,   'YData', py_B2(i) + Prad*circle_y)
    set(B2c, 'XData', px_B2(i) + prad/2*circle_x, 'YData', py_B2(i) + prad/2*circle_y)
    set(textA, 'String', sprintf('Team A: %d / %d',round(score_A(i)),max_score));
    set(textB, 'String', sprintf('Team B: %d / %d',round(score_B(i)),max_score));
    pause(0.04)
    if make_video
        F(i) = getframe(gcf);
    end
end

if make_video
    writerObj = VideoWriter(videofile);
    writerObj.FrameRate = 20;
    % open the video writer
    open(writerObj);
    % write the frames to the video
    for i=1:length(F)
        % convert the image to a frame
        frame = F(i) ;    
        writeVideo(writerObj, frame);
    end
    % close the writer object
    close(writerObj);
end
