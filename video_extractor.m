inputDirectory = 'video-training-data/videos/raw_video';
movementDirectory = 'video-training-data/videos/movement_vectors';
audioDirectory = 'video-training-data/audio/raw_audio';
file_list = dir(fullfile(inputDirectory,'*.mp4'));
for k=1:length(file_list)
    videoFile = fullfile(inputDirectory, file_list(k).name);
    [audioData,audioSampleRate] = audioread(videoFile);
    [~, name, ~] = fileparts(videoFile);
    audioFile = fullfile(audioDirectory, [name, '.wav']);
    audiowrite(audioFile,audioData,audioSampleRate);
    points = face_detector(videoFile);
    face_detector(videoFile);
    movementFile = fullfile(movementDirectory, [name, '.csv']);
    writematrix(points, movementFile);
end