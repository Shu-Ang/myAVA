import json
import os

# 通过循环与判断来找出via的json标注文件
for root, dirs, files in os.walk("./Dataset/choose_frames", topdown=False):
    for file in files:
        #via的json标注文件以_proposal.json结尾
        if "_proposal.json" in file:
            jsonPath = root+'/'+file
            #读取标注文件
            with open(jsonPath, encoding='utf-8') as f:
                line = f.readline()
                viaJson = json.loads(line)
                attributeNum = len(viaJson['attribute'])
                av = {}
                for i in range(attributeNum):
                    av[str(i + 1)] = ''
                for metadata in viaJson['metadata']:
                    #对标注文件中所有av进行修改，av就是当前选中的标注值
                    #下面的1，2，3代表3种多选，如头部、身体、四肢三个部位的行为
                    # 这里的值应动态获取，时间关系，先固定成这样
                    viaJson['metadata'][metadata]["av"] = av
                #修改后的文件名
                fileName , _ = os.path.splitext(file) 
                newname = fileName +'.json'
                with open(root+'/'+newname, 'w') as f2:
                    f2.write(json.dumps(viaJson))
                    f2.close()
                