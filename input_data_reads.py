import os
import random
import numpy as np
import cv2
import time

#获得视频帧数据
def get_frames_data(filename, num_frames_per_clip=6):
    ''' Given a directory containing extracted frames, return a video clip of
    (num_frames_per_clip) consecutive frames as a list of np arrays '''
    # 返回的数组
    ret_arr = []
    for imagename in os.listdir(filename):
        image_name = str(filename) + '/' + imagename
        # 打开图像
        img = cv2.imread(image_name)
        # 将cv2的BGR转为RGB
        img = img[:, :, ::-1]
        # 获得图像数据
        img_data = np.array(img, dtype=np.float32)
        # print(img_data.shape)
        # 将图像数据添加到ret_arr
        ret_arr.append(img_data)
    return ret_arr

#读取视频帧及其标签
def read_clip_and_label(filename, batch_size=8, num_frames_per_clip=6, crop_size=224, shuffle=True):

    # 打开视频文件
    lines = open(filename, 'r')
    # 将文件lines转为list,每行为list类型lines的一个元素
    lines = list(lines)
    # 文件夹的名称先设置为空列表
    read_dirnames = []
    # data设置为空
    data = []
    # label设置为空列表
    label = []
    # 设置batch_index为0
    batch_index = 0

    # np.mean的shape=(16, 112, 112, 3)

    np_mean = np.load('./crop_mean.npy')
    # 如果start_pos没有指定，迫使shuffle = True
    # Forcing shuffle, if start_pos is not specified
    # if start_pos < 0:
    #   shuffle = True
    # 如果shuffle等于True
    if shuffle:
        # 获得视频的索引
        video_indices = list(range(len(lines)))
        # print('before',video_indices)
        # 按照时间设置随机种子
        random.seed(time.time())
        # 将video_indices进行打乱
        random.shuffle(video_indices)
        # print('after',video_indices)
    else:
        # 顺序处理视频
        # Process videos sequentially
        # start_pos
        # 从开始到len(lines)
        video_indices = list(range(0, len(lines)))

    # 变量视频的索引
    for index in video_indices:
        if (batch_index >= batch_size):
            next_batch_start = index
            break
        # 获得第index个视频的名称和类别号
        line = lines[index].strip('\n').split()
        # 获得视频文件名称
        dirname = line[0]
        # 获得视频的类别号，表示属于什么类别
        tmp_label = line[1]
        # print(dirname,tmp_label)
        # 得到视频帧的数据
        tmp_data = get_frames_data(dirname, num_frames_per_clip)
        # print(len(tmp_data))
        # 图像数据img_datas初始化为空列表
        img_datas = []
        # 如果tmp_data的长度不等于0
        if (len(tmp_data) != 0):
            # 遍历tmp_data
            for j in range(len(tmp_data)):
                # 将tmp_data[j]转为np。uint8类型，再从数组转为Image类型
                # img = Image.fromarray(tmp_data[j].astype(np.uint8))
                img=tmp_data[j]
                # 如果图像的宽大于图像的高,
                if (img.shape[0] > img.shape[1]):
                    # 比例
                    scale = float(crop_size) / float(img.shape[1])
                    img=cv2.resize(img, (int(img.shape[0] * scale + 1), crop_size))
                else:
                    # 缩放比例
                    scale = float(crop_size) / float(img.shape[0])
                    # 进行缩放，将短边缩放为crop_size，长边缩放为int(height/width*crop_size+1)
                    img = cv2.resize(img, (crop_size, int(img.shape[1] * scale + 1)))

                # 裁剪的宽的起始位置
                crop_x = int((img.shape[0] - crop_size) / 2)
                # 裁剪的高德起始位置
                crop_y = int((img.shape[1] - crop_size) / 2)
                # 获得中心裁剪后的图像，然后减去第j帧的平均值，shape=[224,224,3]
                img = img[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size, :]-np_mean[j]
                # 将img添加到img_datas中
                img_datas.append(img)
            # img_datas=(num_frames_per_clip,crop_size,crop_size,3]的数据添加到data中

            data.append(img_datas)

            # 将视频的类别号添加到label中
            label.append(int(tmp_label))
            # batch_index累加
            batch_index = batch_index + 1

    # 将data转为numpy数组,类型转为np.float32,图像channles=3
    # 这里np_arr_data的shape=[batch_size,num_frames_per_clip,crop_size,crop_size,channles]
    np_arr_data = np.array(data, dtype=np.float32)
    # np_arr_label的shape=(batch_size,)
    np_arr_label = np.array(label, dtype=np.int32)

    return np_arr_data, np_arr_label
