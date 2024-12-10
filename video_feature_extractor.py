import matlab.engine
import glob
import numpy as np
eng = matlab.engine.start_matlab()
inputDirectory = "video-training-data/videos/raw_video"
movementDirectory = 'video-training-data/videos/movement_vectors'
audioDirectory = 'video-training-data/audio/raw_audio'
f = open((movementDirectory + '/vectors.txt'), 'w')
for video in sorted(glob.glob(f"{inputDirectory}/*.mp4")):
    movement_vectors = []
    points = np.array(eng.face_detector(video))
    for i in range(len(points) - 1):
        move_vec = points[i + 1] - points[i]
        movement_vectors.append(list(move_vec))
    video = video.replace(f"{inputDirectory}\\","").replace('.mp4','')
    print("video:",  video,  "movement_vectors: ", movement_vectors)
    f.write(f"{video}; {movement_vectors}")
    f.write("\n")
f.close()


##  Points for vid 1
##  (4,3)
##  (2,1)
### (1,2
##

######################## Create list which appends all len(points)
####while loop which increments each time a calculation is done
#### for loop (length of each value in first list)

## for i in len(points):
# list = []
##### for j in len(points[i]-1):
############# movement_vec = points[i][j] - points[i][j+1]
### list.append(movement_vec)
# save(list as Filename.txt/csv/etc.)
