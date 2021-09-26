import logging
import os
import argparse
from pathlib import Path
import cv2
import h5py
import torch
import numpy as np
import matplotlib.cm as cm
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

os.sys.path.append('../../')

from src.loftr import LoFTR, default_cfg
from src.utils.parsers import *
from src.utils import io
from demo.utils import *

# pre-defined variables
use_cuda = torch.cuda.is_available()
torch.set_grad_enabled(False)


def load_image_data(image_name, image_path, grayscale, resize, resize_float):
    image = io.read_image(image_path, grayscale)
    raw_image = image
    image = image.astype(np.float32)

    size = image.shape[:2][::-1]
    w, h = size
    w_new, h_new = process_resize(w, h, resize)

    scale_w, scale_h = (w / w_new), (h / h_new)
    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')

    # crop for ResNet
    w_new_crop = (w_new // 8) * 8
    h_new_crop = (h_new // 8) * 8
    image = image[0: h_new_crop, 0: w_new_crop]

    if not grayscale:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW

    # image = image / 255.

    # data = {
    #     'name': image_name,
    #     'image': image,
    #     'original_size': np.array(size),
    # }
    return raw_image, image, scale_w, scale_h


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(
        description='Aachen extract',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dataset', type=Path, default='../../../data/Aachen_v1.1',
        help='Path to the dataset, default: %(default)s')
    parser.add_argument(
        '--input_image_pairs', type=str, default='image_pairs_to_match_v1_1.txt')
    parser.add_argument(
        '--dump_dir', type=str, default=None, help="if set, the matching results will be dump to dump_dir")
    parser.add_argument(
        '--weight', type=str, default='../../../weights/outdoor_ds.ckpt')
    parser.add_argument(
        '--max_length', type=int, default=-1,
        help='Maximum number of pairs to evaluate')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[1200],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    # parser.add_argument(
    #     '--experiment_name', type=str, default='loftr')
    parser.add_argument(
        '--resize_float', action='store_true',
        help='Resize the image after casting uint8 to float')
    parser.add_argument(
        '--image_glob', type=str, nargs='+', default=['*.png', '*.jpg', '*.jpeg'],
        help='Glob if a directory of images is specified')
    parser.add_argument(
        '--no_display', action='store_true',
        help='Do not display images to screen. Useful if running remotely')
    parser.add_argument(
        '--save_video', action='store_true',
        help='Save output (with match visualizations) to a video.')
    parser.add_argument(
        '--top_k', type=int, default=2000, help="The max vis_range (please refer to the code).")
    parser.add_argument(
        '--bottom_k', type=int, default=0, help="The min vis_range (please refer to the code).")
    parser.add_argument('--benchmark', type=bool, default=False)

    opt = parser.parse_args()
    parser.print_help()

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        raise RuntimeError("GPU is required to run this demo.")

    # Initialize LoFTR
    matcher = LoFTR(config=default_cfg)
    matcher.load_state_dict(torch.load(opt.weight)['state_dict'])
    matcher = matcher.eval().to(device=device)

    # Get shortlists for each query image
    # Setup the paths
    dataset = opt.dataset
    images = dataset / 'database_and_query/images_upright/'
    pair_names_fn = dataset / opt.input_image_pairs

    assert pair_names_fn.exists(), pair_names_fn
    pairs = parse_retrieval(pair_names_fn)
    pairs = [(q, r) for q, rs in pairs.items() for r in rs]

    matches_dir = opt.dump_dir
    if matches_dir is None:
        raise ValueError('Provide an dump_dir .')

    matches = Path(matches_dir, f'{Path(opt.input_image_pairs).stem}_{opt.resize[0]}_pred.h5')
    matches.parent.mkdir(exist_ok=True, parents=True)
    skip_pairs = set(io.list_h5_names(matches) if matches.exists() else ())

    timer = AverageTimer()
    vis_range = [opt.bottom_k, opt.top_k]

    for (name0, name1) in tqdm(pairs, smoothing=.1):
        pair = names_to_pair(name0, name1)
        # Avoid to recompute duplicates to save time
        if pair in skip_pairs or names_to_pair(name0, name1) in skip_pairs:
            continue

        image_0, image_1 = images / name0, images / name1
        # print(image_0, image_1)

        raw_img0, src, scale_w_0, scale_h_0 = load_image_data(name0, image_0, True, opt.resize, opt.resize_float)
        raw_img1, tgt, scale_w_1, scale_h_1 = load_image_data(name1, image_1, True, opt.resize, opt.resize_float)

        src_tensor, tgt_tensor = frame2tensor(src, device), frame2tensor(tgt, device)
        data = {'image0': src_tensor, 'image1': tgt_tensor}

        matcher(data)

        total_n_matches = len(data['mkpts0_f'])
        mkpts0 = data['mkpts0_f'].cpu().numpy()[vis_range[0]:vis_range[1]]
        mkpts1 = data['mkpts1_f'].cpu().numpy()[vis_range[0]:vis_range[1]]
        mconf = data['mconf'].cpu().numpy()[vis_range[0]:vis_range[1]]

        mkpts0[:, 0] = (mkpts0[:, 0] + 0.5) * scale_w_0 - 0.5
        mkpts0[:, 1] = (mkpts0[:, 1] + 0.5) * scale_h_0 - 0.5
        mkpts1[:, 0] = (mkpts1[:, 0] + 0.5) * scale_w_1 - 0.5
        mkpts1[:, 1] = (mkpts1[:, 1] + 0.5) * scale_h_1 - 0.5

        # Normalize confidence.
        if len(mconf) > 0:
            conf_vis_min = 0.
            conf_min = mconf.min()
            conf_max = mconf.max()
            mconf = (mconf - conf_vis_min) / (conf_max - conf_vis_min + 1e-5)

        with h5py.File(str(matches), 'a') as fd:
            grp = fd.create_group(pair)
            grp.create_dataset('mkpts0_f', data=mkpts0)
            grp.create_dataset('mkpts1_f', data=mkpts1)
            grp.create_dataset('mconf', data=mconf)

        skip_pairs.add(pair)

        timer.update('forward')
        alpha = 0
        color = cm.jet(mconf, alpha=alpha)

        if not opt.no_display:
            text = [
                f'LoFTR',
                '# Matches (showing/total): {}/{}'.format(len(mkpts0), total_n_matches),
            ]
            small_text = [
                f'Showing matches from {vis_range[0]}:{vis_range[1]}',
                f'Confidence Range: {conf_min:.2f}:{conf_max:.2f}',
                'Image Pair: ' + name0 + ' , ' + name1
            ]
            out = make_matching_plot_fast(
                raw_img0, raw_img1, mkpts0, mkpts1, mkpts0, mkpts1, color, text,
                path=None, show_keypoints=False, small_text=small_text)
            cv2.imshow('LoFTR Matches', out)
            # cv2.waitKey(0)

        timer.update('viz')
        timer.print()

    logging.info('Finished exporting matches.')

    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
