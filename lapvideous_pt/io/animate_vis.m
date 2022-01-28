%% Script to animate a scene generated using lapvideous_render_gen into a gif

% File variable names. Path to files.
file_vars = 'final1_mat_rz_p2l.mat';
file_vars_pred = 'pred_final1_mat_rz_p2l.mat';
gif_name = 'animation1.gif';

% Load
load(file_vars)
load(file_vars_pred)
batch = size(ims);
figure
set(gcf, 'Position', get(0, 'Screensize'));

% Render: iterate over batch, and plot each plane.
for i =1:batch(1)
    % We add +1 to the faces, as in python they are 0-indexed versus
    % 1-indexed in Matlab
    cla
    % Plotting the 3D volume and the vasculature
    handles.a = subplot(2,3,[1 4]); 
    trisurf(liver_faces+1, liver_points(:, 1), liver_points(:, 2), liver_points(:, 3), 'FaceColor', [0.6350 0.0780 0.1840], 'FaceAlpha', 0.1, 'EdgeColor', [0.6350 0.0780 0.1840], 'EdgeAlpha', 0.2);
    hold on
    subplot(2,3,[1 4]);
    trisurf(arteries_faces+1, arteries_points(:, 1), arteries_points(:, 2), arteries_points(:, 3), 'FaceColor', 'y', 'FaceAlpha', 0.4, 'EdgeColor', 'y', 'EdgeAlpha', 0.2);
    hold on
    subplot(2,3,[1 4]);
    trisurf(HV_facecs+1, HV_points(:, 1), HV_points(:, 2), HV_points(:, 3), 'FaceColor', 'g', 'FaceAlpha', 0.4, 'EdgeColor', 'g', 'EdgeAlpha', 0.2);
    hold on
    subplot(2,3,[1 4]);
    trisurf(PV_faces+1, PV_points(:, 1), PV_points(:, 2), PV_points(:, 3), 'FaceColor', 'b', 'FaceAlpha', 0.4, 'EdgeColor', 'b', 'EdgeAlpha', 0.2);
    hold on
    % Scatter the 3D plane intersecting the model
    subplot(2,3,[1 4]);
    scatter3(planes(i, :, 1), planes(i, :, 2), planes(i, :, 3), '.k')
    hold on
    % Set the view of the 3D axis from the position of the camera in space.
    %view(handles.a, [cam_pose(i, :)])%     
    scatter3(cam_pose(i, 1), cam_pose(i, 2), cam_pose(i, 3), 20, 'xc')
    scatter3(lus_pose(i,1), lus_pose(i,2), lus_pose(i,3), 200, '.c', 'Linewidth', 4)
    axis equal
    size(ims)
    subplot(2,3,2); imshow(squeeze(ims(i, :, :, 1:3)/255))
    subplot(2,3,3); imshow(squeeze(ims(i, :, :, 4:6)))
    
    % Scatter the 3D plane intersecting the model
    subplot(2,3,[1 4]); scatter3(planes_pred(i, :, 1), planes_pred(i, :, 2), planes_pred(i, :, 3), '.r')
    hold on
    % Set the view of the 3D axis from the position of the camera in space.
    %view(handles.a, [cam_pose(i, :)])%     
    scatter3(cam_pose_pred(i, 1), cam_pose_pred(i, 2), cam_pose_pred(i, 3), 20, 'xr')
    scatter3(lus_pose_pred(i,1), lus_pose_pred(i,2), lus_pose_pred(i,3), 200, '.r', 'Linewidth', 4)
    axis equal
    subplot(2,3,5); imshow(squeeze(ims_pred(i, :, :, 1:3)/255))
    subplot(2,3,6); imshow(squeeze(ims_pred(i, :, :, 4:6)))
    drawnow
    % Get data for animation
    frame = getframe(gcf);
    img =  frame2im(frame);
    [img,cmap] = rgb2ind(img,256);
    if i == 1
        imwrite(img,cmap,'animation1.gif','gif','LoopCount',Inf,'DelayTime',1);
    else
        imwrite(img,cmap,'animation1.gif','gif','WriteMode','append','DelayTime',0.01);
    end

end
