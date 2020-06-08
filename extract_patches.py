
import openslide
import numpy as np
import glob, os
import argparse
from PIL import Image


def create_patches(slide_fp, level, patch_size, whiteness_limit, blackness_limit, max_faulty_pixels, dest_dir):
    _, filename = os.path.split(slide_fp)
    slide_id, uuid, extension = filename.split('.')
    grade = int(slide_id.split('-')[3][:2])
    if grade>=1 and grade<=9:
        label = "cancer"
    elif grade>=10 and grade<=19:
        label = "normal"
    elif grade >=20:
        label = "control"
    if not os.path.exists(os.path.join(dest_dir, label, slide_id)):
        os.makedirs(os.path.join(dest_dir, label, slide_id))

    slide = openslide.OpenSlide(slide_fp)
    W,H = slide.level_dimensions[level]
    slide_image = slide.read_region(location=(0,0), level=level, size=(W,H)).convert('RGB')
    arr = np.array(slide_image)

    ctr = 0    
    for x in range(0,W,patch_size):
        for y in range(0,H,patch_size):
            patch = arr[x:x+patch_size,y:y+patch_size]
            w,h,_ = patch.shape
            if w<patch_size or h<patch_size:
                continue
            is_white = np.all([patch[:,:,i]>whiteness_limit for i in range(3)], axis=0)
            is_black = np.all([patch[:,:,i]<blackness_limit for i in range(3)], axis=0)
            if np.sum(is_white+is_black)>w*h*max_faulty_pixels:
                continue
            fp = os.path.join(dest_dir, label, slide_id, "{}_X_{}_Y_{}.png".format(slide_id, x, y))
            im_patch = Image.fromarray(patch)
            im_patch.save(fp)
            ctr += 1

    print("Slide {} Done. Slide Dimensions W={} H={}, #Patches={}".format(filename,W,H,ctr), flush=True)


def main():
    parser = argparse.ArgumentParser(description='Process args for patch extraction')
    parser.add_argument("--root_dir", type=str, help="Root directory", default="/ssd_scratch/cvit/MED/KIRC/SLIDES")
    parser.add_argument("--dest_dir", type=str, help="Destination directory", default="/ssd_scratch/cvit/MED/KIRC/PATCHES")
    parser.add_argument("--level", type=int, help="0 means 40x, 1 means 20x", default=1)
    parser.add_argument("--patch_size", type=int, default=512, help="Patch Size")
    parser.add_argument("--whiteness_limit", type=int, default=210, help="Whiteness Limit")
    parser.add_argument("--blackness_limit", type=int, default=5, help="Blackness Limit")
    parser.add_argument("--max_faulty_pixels", type=float, default=0.6, help="Max allowed fraction of only B/W pixels")

    args = parser.parse_args()

    ## change this if your storage format is different
    file_pattern = "{}/subset_*/*/*.svs".format(args.root_dir)
    # file_pattern = "{}/*/*.svs".format(args.root_dir)
    slide_files = glob.glob(file_pattern)

    for slide_file in slide_files:
        create_patches(slide_file,args.level,args.patch_size,args.whiteness_limit,\
                    args.blackness_limit, args.max_faulty_pixels, args.dest_dir)


if __name__=="__main__":
    main()
