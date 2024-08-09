# 1 Dataset‘s folder structure 数据集文件结构
Dataset
- annotations
    + train.csv
    +  val.csv
    + actionlist.pbtxt
- rawframes
# 2 环境准备
``` bash
Pytorch 1.8.0，python 3.8，CUDA 11.1.1
# ffmpeg用于处理视频
conda install x264 ffmpeg -c conda-forge -y
# 其余需要的库
pip install -r ./yolovDeepsort/requirements.txt
pip install opencv-python-headless==4.1.2.30
# 下载预训练模型 （如果速度太慢可以在本地下载上传服务器到指定目录）
wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt -O ./yolovDeepsort/yolov5/yolov5s.pt 
mkdir -p ~/.config/Ultralytics/
wget  https://ultralytics.com/assets/Arial.ttf -O ~/.config/Ultralytics/Arial.ttf
wget https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6 -O ./yolovDeepsort/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7 
```


# 3 数据集视频准备
将视频上传至`./Dataset/videos`目录下，一次可以对多个视频进行处理
![image](https://img-blog.csdnimg.cn/1f996811ec164f08b21f04e42220601a.png)
# 4 对视频进行裁剪、抽帧，并使用yolov5检测
**以下命令都在ava目录下执行**
```
bash ./step1.sh
```
之后将`./Dataset/choose_frames.zip`下载到本地并解压
# 5 使用via标注
下载via工具https://www.robots.ox.ac.uk/~vgg/software/via/downloads/via3/via-3.0.13.zip

点击标注工具中的： via_image_annotator.html<br>

![image](https://img-blog.csdnimg.cn/fec0e87d18ab48c2af8299791a1e71af.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQ1Yt5p2o5biG,size_18,color_FFFFFF,t_70,g_se,x_16)

下图是via的界面，1代表添加图片，2代表添加标注文件<br>

![image](https://img-blog.csdnimg.cn/6c896dd36f284f2286867510c705a7de.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQ1Yt5p2o5biG,size_20,color_FFFFFF,t_70,g_se,x_16)

导入图片，打开标注文件（注意，打开x_x_proposal_s.json），最后结果：<br>
![image](https://img-blog.csdnimg.cn/ba44be0e5d454a2ba063e363b179daea.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQ1Yt5p2o5biG,size_20,color_FFFFFF,t_70,g_se,x_16)

标注完成后，导出json并命名为videoname_finish，之后上传到服务器对应目录下`./Dataset/choose_frames/videoname/`

# 6 使用deepsort跟踪对象得到personID，并整合生成数据集
```
bash step2.sh
```
之后会在`./Dataset/annotations/`目录下得到`train.csv`和`train_without.csv`

其中`train.csv`和`val.csv`结构如下：
|video_name|sec|x1|y1|x2|y2|actionID|personID|
| ---------|---|--|--|--|--|--------|--------|   

**注意，执行`clean1.sh`和`clean2.sh`可以将对应步骤的中间输出删除。开始标注之前务必执行这两个脚本以删除上次标注的中间输出。**

