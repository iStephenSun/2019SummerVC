import os
from PIL import Image


# 定义图像拼接函数
def image_compose(images_path, image_size, image_row, image_column, image_save_path):
    # 小图类型
    IMAGES_FORMAT = ['.jpg', '.JPG']

    # 获取图片集地址下的所有图片名称
    image_names = [name for name in os.listdir(images_path) for item in IMAGES_FORMAT if os.path.splitext(name)[1] == item]

    # 简单的对于参数的设定和实际图片集的大小进行数量判断
    if len(image_names) % (image_row * image_column) != 0:
        raise ValueError("合成图片的参数和要求的数量不能匹配！")

    # 需要合成的四格图的张数
    num = int(len(image_names) / (image_row * image_column))

    for i in range(1, num + 1):
        # 创建一个新图
        to_image = Image.new('RGB', (image_column * image_size, image_row * image_size))

        # 循环遍历，把每张图片按顺序粘贴到对应位置上
        for y in range(1, image_row + 1):
            for x in range(1, image_column + 1):
                from_image = Image.open(
                    images_path + image_names[(4 * (i - 1)) + (image_column * (y - 1) + x - 1)]).resize(
                    (image_size, image_size), Image.ANTIALIAS)
                to_image.paste(from_image, ((x - 1) * image_size, (y - 1) * image_size))

        # 保存新图
        new_path = image_save_path + str(i) + ".jpg"
        to_image.save(new_path)
