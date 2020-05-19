import os
import pascal_voc_io
import parseJson
import argparse

parser = argparse.ArgumentParser(description="BDD100K")
parser.add_argument('--srcDir', default=None)
parser.add_argument('--outputRoot', default=None)

def main():
    args = parser.parse_args()
    srcDir = args.srcDir
    outputRoot = args.outputRoot

    dstDirTrain = os.path.join(outputRoot, "VOC2007/Annotations/")
    dstDirDayTime = os.path.join(outputRoot, "daytime", "VOC2007/Annotations")
    dstDirDusk = os.path.join(outputRoot, "dusk", "VOC2007/Annotations")
    dstDirNight = os.path.join(outputRoot, "night", "VOC2007/Annotations")
    i = 1
    # os.walk()
    # dirName是你所要遍历的目录的地址, 返回的是一个三元组(root,dirs,files)
    # root所指的是当前正在遍历的这个文件夹的本身的地址
    # dirs是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
    # files 同样是 list , 内容是该文件夹中所有的文件(不包括子目录)
    for dirpath, dirnames, filenames in os.walk(srcDir):
#        print(dirpath, dirnames, filenames)
        for filepath in filenames:
            fileName = os.path.join(dirpath,filepath)
            # print(fileName)
            print("processing: {}, {}".format(i, fileName))
            i = i + 1
            xmlFileName = filepath[:-5] # remove ".json" 5 character
            # 解析该json文件，返回一个列表的列表，存储了一个json文件里面的所有方框坐标及其所属的类
            objs, timeofday = parseJson.parseJson(str(fileName)) 
            # 如果存在检测对象，创建一个与该json文件具有相同名的VOC格式的xml文件
            if len(objs):
                if timeofday == 'daytime':
                    tmp = pascal_voc_io.PascalVocWriter(dstDirDayTime, xmlFileName, (720,1280,3), fileName)
                    for obj in objs:
                        tmp.addBndBox(obj[0],obj[1],obj[2],obj[3],obj[4])
                    tmp.save()
                elif timeofday == 'dawn/dusk':
                    tmp = pascal_voc_io.PascalVocWriter(dstDirDusk, xmlFileName, (720,1280,3), fileName)
                    for obj in objs:
                        tmp.addBndBox(obj[0],obj[1],obj[2],obj[3],obj[4])
                    tmp.save()
                else:
                    tmp = pascal_voc_io.PascalVocWriter(dstDirNight, xmlFileName, (720,1280,3), fileName)
                    objsTrue = objs[:-1]
                    for obj in objsTrue:
                        tmp.addBndBox(obj[0],obj[1],obj[2],obj[3],obj[4])
                    tmp.save()
                # tmp = pascal_voc_io.PascalVocWriter(dstDirTrain, xmlFileName, (720,1280,3), fileName)
                # objsTrue = objs[:-1]
                # for obj in objsTrue:
                #     tmp.addBndBox(obj[0],obj[1],obj[2],obj[3],obj[4])
                # tmp.save()
            else:
                print(fileName)

if __name__ == '__main__':
    # test
    # these paths should be your own path
#    srcDir = '/media/xavier/SSD256/global_datasets/BDD00K/bdd100k/labels/100k/val'
#    dstDirDayTime = '/media/xavier/SSD256/global_datasets/BDD00K/bdd100k/Annotations/val'

    
    main()
