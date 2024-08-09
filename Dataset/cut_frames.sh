IN_DATA_DIR="./Dataset/videos"
OUT_DATA_DIR="./Dataset/frames"
DATA_DIR="./Dataset/rawframes"

videos=$(ls $IN_DATA_DIR)
for video in $videos; do
    video_name=$(echo $video | cut -d. -f1).$(echo $video | cut -d. -f2)
    video_path=$IN_DATA_DIR/$video
    frames_dir=${OUT_DATA_DIR}/$video_name
    rawframes_dir=${DATA_DIR}/$video_name
    
    if [ ! -d $rawframes_dir ] && [ ! -d $frames_dir ]; then
        mkdir -p "${frames_dir}"
        out_name="${frames_dir}/${video_name}_%06d.jpg"
        ffmpeg -i "${video_path}" -r 30 -q:v 1 "${out_name}"
    fi
done