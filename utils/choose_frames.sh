VIDEO_DIR=$1
IN_DATA_DIR="./Dataset/frames"
ALL_OUT_DATA_DIR="./Dataset/choose_frames_all"
OUT_DATA_DIR="./Dataset/choose_frames"


videos=$(ls $IN_DATA_DIR)
for video in $videos; do
    if [ ! -d "$OUT_DATA_DIR/$video" ]; then
        mkdir $OUT_DATA_DIR/$video
        video_path=$VIDEO_DIR/$video.mp4
        
        duration=$(ffmpeg -i "$video_path" 2>&1 | grep -oP '(?<=Duration: )[^,]*')
        duration=$(echo $duration | awk -F: '{printf "%d:%d:%d", $1, $2, $3}')
        hours=$(echo $duration | cut -d: -f1 )
        minutes=$(echo $duration | cut -d: -f2 ) 
        seconds=$(echo $duration | cut -d: -f3 | cut -d. -f1)
        total_seconds=$(( $hours * 3600 + $minutes * 60 + $seconds ))

        nameTemplate="${video}_%06d.jpg"
        if (( $total_seconds > 4 )); then
            for ((i=1; i<=(($total_seconds * 30 - 89));i+=30));do
                imageName=$(printf "$nameTemplate" $i)
                imagePath=$IN_DATA_DIR/$video/$imageName
                if [ -f "$imagePath" ]; then
                    cp $imagePath $ALL_OUT_DATA_DIR
                    cp $imagePath $OUT_DATA_DIR/$video
                fi
            done
        fi
    fi

done