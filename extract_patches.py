
import openslide
import numpy as np
import glob, os
import argparse
from PIL import Image
import cv2
import time
from multiprocessing import Pool
import multiprocessing
import itertools


def get_connected_components(img):
    img = cv2.cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
    num_labels, labels_im = cv2.connectedComponents(img)
    return num_labels,labels_im


def create_patches(slide_fp, level, patch_size, whiteness_limit, blackness_limit, max_faulty_pixels, min_conn_comp, dest_dir, extras_dir):
    try:
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
        W,H = slide.level_dimensions[0]
        patch_size_upscaled = 4*patch_size
        W,H = W-W%patch_size_upscaled,H-H%patch_size_upscaled

        # im_slide = np.full((W//4,H//4,3),240,dtype=np.uint8)

        ctr,ctr_rej = 0,0
        for x in range(0,W//4,patch_size):
            for y in range(0,H//4,patch_size):
                patch = slide.read_region(location=(4*x,4*y), level=0, \
                        size=(patch_size_upscaled,patch_size_upscaled)).convert('RGB')
                patch = patch.resize((patch_size,patch_size))
                patch_arr = np.swapaxes(np.array(patch),1,0)

                is_white = np.all([patch_arr[:,:,i]>whiteness_limit for i in range(3)], axis=0)
                is_black = np.all([patch_arr[:,:,i]<blackness_limit for i in range(3)], axis=0)
                num_labels, labels_im = get_connected_components(patch_arr)

                if np.sum(is_white+is_black)>patch_size*patch_size*max_faulty_pixels and num_labels<min_conn_comp:
                    ctr_rej += 1
                    continue

                fp = os.path.join(dest_dir, label, slide_id, "{}_X_{}_Y_{}.png".format(slide_id, x, y))
                patch.save(fp)

                # im_slide[x:x+patch_size,y:y+patch_size,:] = patch_arr
                ctr += 1

        # Image.fromarray(np.swapaxes(im_slide,1,0)).save(os.path.join(extras_dir, "full_slides", "{}.png".format(slide_id)))
        thumbnail = slide.get_thumbnail((1024,1024)) # Gets automatically resized as per aspect ratio
        thumbnail.save(os.path.join(extras_dir,"thumbnails","{}.jpg".format(slide_id)),format="jpeg")
        print("[Time] {} Slide {} Grade {} Dimensions W={} H={}, #Patches={} acc {} rej total {}".format(time.strftime("%d-%b-%Y %H:%M:%S"),filename,grade,W,H,ctr,ctr_rej,ctr+ctr_rej), flush=True)

    except:
        print("[Time] {} {} Not Done.".format(time.strftime("%d-%b-%Y %H:%M:%S"),slide_fp))



def main():
    parser = argparse.ArgumentParser(description='Process args for patch extraction')
    parser.add_argument("--root_dir", type=str, help="Root directory", default="/ssd_scratch/cvit/MED/KIRC/SLIDES")
    parser.add_argument("--dest_dir", type=str, help="Destination directory", default="/ssd_scratch/cvit/MED/KIRC/PATCHES")
    parser.add_argument("--extras_dir", type=str, help="Extras directory", default="/ssd_scratch/cvit/MED/KIRC/EXTRAS")
    parser.add_argument("--level", type=int, help="0 means 40x, 1 means 20x", default=1)
    parser.add_argument("--patch_size", type=int, default=512, help="Patch Size")
    parser.add_argument("--whiteness_limit", type=int, default=210, help="Whiteness Limit")
    parser.add_argument("--blackness_limit", type=int, default=5, help="Blackness Limit")
    parser.add_argument("--max_faulty_pixels", type=float, default=0.6, help="Max allowed fraction of only B/W pixels")
    parser.add_argument("--min_conn_comp", type=int, default=10, help="Min allowed number of connected components")

    args = parser.parse_args()
    os.makedirs(os.path.join(args.extras_dir, "thumbnails"), exist_ok=True)

    ## change this if your storage format is different
    file_pattern = "{}/subset_*/*/*.svs".format(args.root_dir)
    slide_files = glob.glob(file_pattern)

    pool = Pool(multiprocessing.cpu_count())
    P = itertools.product(slide_files, [args.level],[args.patch_size],[args.whiteness_limit],\
    [args.blackness_limit], [args.max_faulty_pixels], [args.min_conn_comp], [args.dest_dir], [args.extras_dir])

    pool.starmap(create_patches, P)



if __name__=="__main__":
    main()
