
import argparse
from PIL import Image
import numpy as np
import glob
import re
import cv2

# PATCH_SIZE = 512*4
PATCH_SIZE = 512

parser = argparse.ArgumentParser(description='Process args for Visualization')
parser.add_argument("--img_dir", type=str, required=True)
parser.add_argument("--jpg_path", type=str, required=True)

def recreate_slide(img_dir, jpg_path):
    img_files = glob.glob("{}/*.png".format(img_dir))
    top_left_x_s = [int(re.search("_X_\d+", f).group(0).split('_')[-1]) for f in img_files]
    top_left_y_s = [int(re.search("_Y_\d+", f).group(0).split('_')[-1]) for f in img_files]
    X_max = max(top_left_x_s)
    Y_max = max(top_left_y_s)
    arr = np.full((X_max+2*PATCH_SIZE, Y_max+2*PATCH_SIZE, 3), 255, dtype=np.uint8)
    for i, img_file in enumerate(img_files):
        tl_x, tl_y = top_left_x_s[i], top_left_y_s[i]
        br_x, br_y = tl_x+PATCH_SIZE, tl_y+PATCH_SIZE
        # img = Image.open(img_file).resize((PATCH_SIZE, PATCH_SIZE), resample=Image.NEAREST)
        img = cv2.imread(img_file)
        arr[tl_x:br_x, tl_y:br_y] = np.array(img)

    slide_img = cv2.resize(arr, dsize=(1024,1024))
    cv2.imwrite(jpg_path, slide_img)


def main():
    args = parser.parse_args()
    img_dir = args.img_dir
    jpg_path = args.jpg_path
    recreate_slide(img_dir, jpg_path)

if __name__=="__main__":
    main()
