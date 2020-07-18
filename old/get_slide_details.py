
import openslide
import argparse
import os, glob

parser = argparse.ArgumentParser(description='Process args for Slide Data Analysis')
parser.add_argument("--root_dir", type=str, required=True, help="Dir inside which subset folders reside")
parser.add_argument("--detail_file", type=str, required=True, help="txt file to write details")


def main():
    args = parser.parse_args()
    root_dir = args.root_dir
    detail_file = args.detail_file
    svs_files = glob.glob("{}/*/*/*.svs".format(root_dir))
    writer = open(detail_file, 'a+')
    for svs_file in svs_files:
        try:
            slide = openslide.OpenSlide(svs_file)
            w, h = slide.dimensions
            writer.write("{}\t{}\t{}\n".format(svs_file.split('/')[-1][:-4], w, h))
            writer.flush()
            slide.close()
        except:
            print("Didnt work for {}".format(svs_file), flush=True)
    writer.close()


if __name__=="__main__":
    main()
