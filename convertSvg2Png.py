import os
import sys
import glob 
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Convert SVG to PNG")
parser.add_argument('--svg_dir', dest="svg_dir")
parser.add_argument('--target_dir', dest="target_dir")
args = parser.parse_args()


def makedir(path):
	if not os.path.exists(path):
		os.makedirs(path)

makedir(args.target_dir)
for source_path in tqdm(glob.glob(os.path.join(args.svg_dir,"*.svg"))):
	base_name = os.path.basename(source_path).replace(".svg", ".jpg")
	target_path = os.path.join(args.target_dir, base_name)
	os.system("cairosvg {} -o {}".format(source_path, target_path))




