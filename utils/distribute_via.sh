split=$1

FRAME_DIR=Dataset/choose_frames
VIA_DIR=Dataset/${split}_finish

files=$(ls $VIA_DIR)
for file in $files; do
    video=$(echo "$file" | sed 's/_finish\.json$//')
    video_path=$FRAME_DIR/$video
    via_path=$VIA_DIR/$file
    if [ -d $video_path ];then
        mv $via_path $video_path
    fi
done