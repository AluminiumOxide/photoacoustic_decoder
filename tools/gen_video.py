import cv2
import os



if __name__ == '__main__':
    os.chdir('..')  # 先回到根目录，不然会出问题

    # 图片序列所在的文件夹路径
    image_folder = './lyp11_2/'

    # 视频文件的输出路径和名称
    video_name = image_folder + 'output_video.mp4'

    # 获取图片序列中的所有图片文件
    images_list = []
    images_list_index = []
    for img_name in os.listdir(image_folder):
        if len(img_name.split('_')) == 3:
            img_type = img_name.split('_')[1]
        if img_name.endswith(".png") and (img_type == 'p0'):
            img_index = int(img_name.split('_')[-1].split('.')[0])
            images_list.append(img_name)
            images_list_index.append(img_index)

    print(images_list)
    print(images_list_index)
    # 排序
    paired_data = zip(images_list_index, images_list)
    sorted_data = sorted(paired_data, key=lambda x: x[0])  # 使用sorted函数按照索引排序
    images_list = [item[1] for item in sorted_data]  # 提取排序后的字符串数组

    # 获取第一张图片的宽度和高度，用于设置视频的分辨率
    frame = cv2.imread(os.path.join(image_folder, images_list[0]))
    height, width, layers = frame.shape

    # 使用OpenCV的VideoWriter对象创建视频文件
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编解码器，可以根据需要更改
    video = cv2.VideoWriter(video_name, fourcc, 30, (width, height))  # 参数依次是输出文件名、编解码器、帧速率、分辨率

    # 逐一将图片添加到视频中
    for image in images_list:
        img = cv2.imread(os.path.join(image_folder, image))
        video.write(img)

    # 完成后释放VideoWriter对象
    video.release()

    print(f'视频已创建：{video_name}')