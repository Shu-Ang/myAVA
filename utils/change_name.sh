videos_dir=Dataset/val_videos
videos=$(ls $videos_dir)

for video in $videos;do
    video_path=$videos_dir/$video
    video_name=$(basename $video .mp4)
    new_video_path=$videos_dir/${video_name}_val.mp4
    mv $video_path $new_video_path
done