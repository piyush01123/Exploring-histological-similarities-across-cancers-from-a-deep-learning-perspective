
import openslide
import argparse
import numpy as np
import threading
import os

parser = argparse.ArgumentParser(description='Process args for patch extraction')
parser.add_argument("--svs_path", type=str, required=True, help="SVS path")
parser.add_argument("--dest_dir", type=str, required=True, help="Destination directory of PNG files")
parser.add_argument("--level", type=int, default=0, help="0 means 40x, 1 means 20x")
parser.add_argument("--patch_size", type=int, default=512, help="Patch Size")

parser.add_argument("--whiteness_limit", type=int, default=210, help="Whiteness Limit")
parser.add_argument("--blackness_limit", type=int, default=5, help="Blackness Limit")
parser.add_argument("--max_faulty_pixels", type=int, default=160000, help="Max allowed # of only B/W pixels")


def create_patches(slide_file, level, patch_size, whiteness_limit, blackness_limit, max_faulty_pixels, dest_dir):
    slide_id = slide_file.split('/')[-1][:-4]
    slide = openslide.OpenSlide(slide_file)
    width, height = slide.dimensions

    for x in range(0, width, patch_size):
        for y in range(0, height, patch_size):
            print(x, y)
            patch = slide.read_region(location=(x,y), level=level, size=(patch_size, patch_size)).convert('RGB')
            arr = np.array(patch)
            is_white = np.all([arr[:,:,i]>whiteness_limit for i in range(3)], axis=0)
            is_black = np.all([arr[:,:,i]<blackness_limit for i in range(3)], axis=0)
            flag = "useful" if not np.sum(is_white)+np.sum(is_black)>max_faulty_pixels else "useless"
            patch.save(os.path.join(dest_dir, "{}_X_{}_Y_{}_{}.png".format(slide_id, x, y, flag)))
    slide.close()
    print("Slide {} Done.".format(slide_file), flush=True)


    # def reject_or_save_patch(x, y):
    #     patch = slide.read_region(location=(x,y), level=level, size=(patch_size, patch_size)).convert('RGB')
    #     arr = np.array(patch)
    #     is_white = np.all([arr[:,:,i]>whiteness_limit for i in range(3)], axis=0)
    #     is_black = np.all([arr[:,:,i]<blackness_limit for i in range(3)], axis=0)
    #     flag = "useful" if not np.sum(is_white)+np.sum(is_black)>max_faulty_pixels else "useless"
    #     patch.save(os.path.join(dest_dir, "{}_X_{}_Y_{}_{}.png".format(slide_id, x, y, flag)))
    #
    # threads=[]
    # for x in range(0, width, patch_size):
    #     for y in range(0, height, patch_size):
    #         t = threading.Thread(target=reject_or_save_patch, args=(x, y))
    #         threads.append(t)
    # thread_groups = [threads[i*5:(i+1)*5] for i in range(len(threads)//5+1)]
    # for grp in thread_groups:
    #     for t in grp:
    #         t.start()
    #     for t in grp:
    #         t.join()
    # slide.close()
    # print("Slide {} Done.".format(slide_file), flush=True)


def main():
    args = parser.parse_args()
    slide_file = args.svs_path
    dest_dir = args.dest_dir
    level = args.level
    patch_size = args.patch_size
    whiteness_limit = args.whiteness_limit
    blackness_limit = args.blackness_limit
    max_faulty_pixels = args.max_faulty_pixels
    create_patches(slide_file, level, patch_size, whiteness_limit, blackness_limit, max_faulty_pixels, dest_dir)

if __name__=="__main__":
    main()
