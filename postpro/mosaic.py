
import argparse
from PIL import Image
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--num_img_on_width", type=int, required=True)
    parser.add_argument("--num_img_on_height", type=int, required=True)
    parser.add_argument("--margin_on_width", type=int, default=0)
    parser.add_argument("--margin_on_height", type=int, default=0)
    parser.add_argument("--outfile", type=str, required=True)
    args = parser.parse_args()
    print(args)
    files = os.listdir(args.folder)
    # files = sorted(files, key=lambda x: int(x.split('.jpg')[0]))
    # print(files)
    imgs = [Image.open(os.path.join(args.folder, fp)) for  fp in files]
    M,N = args.num_img_on_width, args.num_img_on_height
    w,h = imgs[0].size
    mw, mh = args.margin_on_width, args.margin_on_height
    A = np.full(( (h-2*mh)*N, (w-2*mw)*M, 3 ), 255)
    for i in range(N):
        for j in range(M):
            try:
                arr = np.array(imgs[i*M+j])[mh:-mh, mw:-mw, :]
                A[i*(h-2*mh):(i+1)*(h-2*mh), j*(w-2*mw):(j+1)*(w-2*mw), :] = arr
            except:
                pass
    Image.fromarray(A.astype(np.uint8)).save(args.outfile)

if __name__=="__main__":
    main()
