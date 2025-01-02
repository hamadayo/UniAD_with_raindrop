from PIL import Image

image_path = "/home/yoshi-22/UniAD/ROLE/nuscenes/samples/CAM_BACK/n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151604037558.jpg"

with Image.open(image_path) as img:
    width, height = img.size
    print(f"Width: {width}, Height: {height}")
