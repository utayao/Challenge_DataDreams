import os
import sys
import glob
import pdb
sys.path.append('../')
from utils import utils
from step_1.evaluator import Evaluator


def main(argv=None):
    pdb.set_trace()
    images = sys.argv[1]
    Eval_data = Evaluator(sys.argv[2])
    if not os.path.isfile(images):
        images = glob.glob(os.path.join(images, '*.tiff'))
    else:
        images = [images]
    images = utils.read_images_arr(images)
    heat_maps = []
    for image in images:
        heat_map = Eval_data.build_heatmap(image, stride=224)
        heat_maps.append(heat_map)
        utils.display(heat_map)

main()
