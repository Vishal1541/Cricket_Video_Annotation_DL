## trim.txt format
h1 m1 s1 h2 m2 s2

## Ground Truth
Format: <PathToImage> 2 <Xa1> <Ya1> <Xa2> <Ya2> <IDa> <Xb1> <Yb1> <Xb2> <Yb2> <IDb>

## How to run
- Put video as `a.mp4` in the `../videos` folder.
- Fill the `trim.txt`.
- Run the `trim_video.ipynb`.
- Run the `video_to_frames.ipynb`.
- Fill the `groundtruth.txt`.
- Run the `crop_players.ipynb`.