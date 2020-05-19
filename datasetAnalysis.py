# -*- coding: utf8 -*-
import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sci
from sklearn.cluster import KMeans


parser = argparse.ArgumentParser(description="BDD100K")
parser.add_argument('--srcDir', default='/Users/apple/Desktop/bdd100k/labels/100k/train/')
parser.add_argument('--outputRoot', default=None)
categorys = ['car', 'bus', 'person', 'bike', 'truck', 'motor', 'train', 'rider', 'traffic sign', 'traffic light']

def main():
    args = parser.parse_args()
    srcDir = args.srcDir
    # outputRoot = args.outputRoot

    # dstDirTrain = os.path.join(outputRoot, "VOC2007/Annotations/")
    # dstDirDayTime = os.path.join(outputRoot, "daytime", "VOC2007/Annotations")
    # dstDirDusk = os.path.join(outputRoot, "dusk", "VOC2007/Annotations")
    # dstDirNight = os.path.join(outputRoot, "night", "VOC2007/Annotations")
    i = 1
    # os.walk()
    # dirName是你所要遍历的目录的地址, 返回的是一个三元组(root,dirs,files)
    # root所指的是当前正在遍历的这个文件夹的本身的地址
    # dirs是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
    # files 同样是 list , 内容是该文件夹中所有的文件(不包括子目录)
    w, h = [], []
    for dirpath, dirnames, filenames in os.walk(srcDir):
        for filepath in filenames:
            fileName = os.path.join(dirpath,filepath)
            # print(fileName)
            print("processing: {}, {}".format(i, fileName))
            i = i + 1
            xmlFileName = filepath[:-5] # remove ".json" 5 character
            # 解析该json文件，返回一个列表的列表，存储了一个json文件里面的所有方框坐标及其所属的类
            objs = parseJson(str(fileName)) 
            for obj in objs:
                w.append(obj[0])
                h.append(obj[1])
            
    w=np.asarray(w)
    h=np.asarray(h)

    x=[h, w]
    x=np.asarray(x)
    x=x.transpose()
    data = {
        'w': w,
        'h': h
    }
    kmeans3 = KMeans(n_clusters=9)
    kmeans3.fit(x)
    y_kmeans3 = kmeans3.predict(x)
    centers3 = kmeans3.cluster_centers_

    yolo_anchor_average=[]
    for ind in range (9):
        yolo_anchor_average.append(np.mean(x[y_kmeans3==ind],axis=0))
    yolo_anchor_average=np.array(yolo_anchor_average)

    yolo_anchor_average=np.array(yolo_anchor_average)

    plt.scatter(x[:, 0], x[:, 1], c=y_kmeans3, s=2, cmap='viridis')
    plt.scatter(yolo_anchor_average[:, 0], yolo_anchor_average[:, 1], c='red', s=50);
    yoloV3anchors = yolo_anchor_average

    plt.axis([0,1280,0,720])
    plt.savefig('Kmeans++bdd100k.jpg')
    plt.show()  
    # frame = pd.DataFrame(data)
    # fig = sns.jointplot(x=frame['w'], y=frame['h'], #设置xy轴，显示columns名称
    #           data = frame,  #设置数据
    #         #   color = 'b', #设置颜色
    #         #   s = 50, edgecolor = 'w', linewidth = 1,#设置散点大小、边缘颜色及宽度(只针对scatter)
    #         #   stat_func=sci.pearsonr,
    #         #   kind = 'scatter',#设置类型：'scatter','reg','resid','kde','hex'
    #         #   #stat_func=<function pearsonr>,
    #            # space = 100, #设置散点图和布局图的间距
    #         #   # size = 8, #图表大小(自动调整为正方形))
    #         #   ratio = 5, #散点图与布局图高度比，整型
    #         #   marginal_kws = dict(bins=15, rug =True), #设置柱状图箱数，是否设置rug
    #         )
    # fig.savefig('dataframe.jpg')
    
def parseJson(jsonFile):
    '''
      params:
        jsonFile -- BDD00K数据集的一个json标签文件
      return:
        返回一个列表的列表，存储了一个json文件里面的方框坐标及其所属的类，
        形如：[[325, 342, 376, 384, 'car'], [245, 333, 336, 389, 'car']]
    '''
    objs = []
    obj = [] # [h, w, catagory]
    f = open(jsonFile)
    info = json.load(f)
    objects = info['frames'][0]['objects']
    timeofday = info['attributes']['timeofday']
    for i in objects:
        if(i['category'] in categorys):
            y2 = float(i['box2d']['y2'])
            y1 = float(i['box2d']['y1'])
            x2 = float(i['box2d']['x2'])
            x1 = float(i['box2d']['x1'])
            h = y2 - y1
            w = x2 - x1

            # if (y2 - y1) * (x2 - x1) < 
            obj.append(h)
            obj.append(w)
            # obj.append(y1)
            # obj.append(y2)

            obj.append(i['category'])
            objs.append(obj)
            obj = []
    return objs
if __name__ == '__main__':
    # test
    # these paths should be your own path
#    srcDir = '/media/xavier/SSD256/global_datasets/BDD00K/bdd100k/labels/100k/val'
#    dstDirDayTime = '/media/xavier/SSD256/global_datasets/BDD00K/bdd100k/Annotations/val'

    
    main()
