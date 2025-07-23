import os
import argparse
import time
import numpy as np
import cv2
import torch
import torchvision.transforms.functional as tvf
from models import FastDVDnet
from accelerate import Accelerator
from einops import rearrange
from tqdm import tqdm

NUM_IN_FR_EXT = 5
NUM_HALF_EXT = NUM_IN_FR_EXT // 2

def parse_args():
    parser = argparse.ArgumentParser(description="FastDVDnet Inference")
    parser.add_argument('--input_dir', type=str, required=True, help='Path to input directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--checkpoint', type=str, default='pretrained_models/model_export.pt', help='Path to model checkpoint')
    parser.add_argument('--noise_sigma', type=float, required=True, help='Noise standard deviation')
    parser.add_argument('--tile_sz', type=int, default=512, help='Tile size for inference')
    parser.add_argument('--tile_overlap', type=int, default=20, help='Tile overlap for inference')
    return parser.parse_args()

def inference_frame_by_tiles(model, input_tensor, noise_map,
                             tile_sz, tile_overlap, batch_size=4):
    # to tiles
    B = input_tensor.shape[0]
    input_tiles = []
    input_bboxes = []
    for i in range(0, input_tensor.shape[2], tile_sz - tile_overlap):
        for j in range(0, input_tensor.shape[3], tile_sz - tile_overlap):
            i = min(i, input_tensor.shape[2] - tile_sz)
            j = min(j, input_tensor.shape[3] - tile_sz)
            input_tiles.append(input_tensor[..., i:i + tile_sz, j:j + tile_sz])
            input_bboxes.append((i, j, i+tile_sz, j+tile_sz))
    input_tiles = torch.concat(input_tiles, dim=0)

    # Inference
    output_tiles = []
    with torch.no_grad():
        for tiles in input_tiles.split(batch_size):
            output = model(tiles, noise_map.expand(tiles.shape[0], -1, -1, -1))
            output_tiles.append(output)
    output_tiles = torch.concat(output_tiles, dim=0)

    # Post-processing
    output_val = torch.zeros_like(input_tensor[:, :3, ...])
    output_w = torch.zeros_like(input_tensor[:, :3, ...])
    for output_tile, bbox in zip(output_tiles, input_bboxes):
        i1, j1, i2, j2 = bbox
        output_val[:, :, i1:i2, j1:j2].add_(output_tile)
        output_w[:, :, i1:i2, j1:j2].add_(1)

    return output_val.div_(output_w)

def inference_frame_by_idx(data_list, cur_i, model, noise_map, tile_sz, tile_overlap):
    # Prepare input tensors
    last_idx = len(data_list) - 1
    input_frames = []
    for i in range(NUM_IN_FR_EXT):
        relidx = abs(cur_i + i - NUM_HALF_EXT) # handle border conditions, reflect
        if relidx > last_idx:
            relidx = last_idx - relidx + last_idx
        input_frames.append(data_list[relidx]['img'])
    input_tensor = torch.stack(input_frames, dim=0).cuda()
    input_tensor = rearrange(input_tensor, 'n c h w -> 1 (n c) h w')  # Reshape to [1, num_frames*C, H, W]

    # Inference
    output = inference_frame_by_tiles(model, input_tensor, noise_map, tile_sz, tile_overlap)

    # Save output
    output = output.clamp(0, 1).squeeze().cpu().numpy().transpose(1, 2, 0)
    output = (output * 255).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(args.output_dir, f"{data_list[cur_i]['fn']}"), output)

def main(args):
    # model
    model = FastDVDnet(num_input_frames=NUM_IN_FR_EXT)

    state_temp_dict = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(state_temp_dict)
    model.eval()

    # accelerate
    accelerate = Accelerator()
    model = accelerate.prepare(model)

    noisestd = torch.FloatTensor([args.noise_sigma / 255.0]).cuda()
    tile_sz = args.tile_sz
    tile_overlap = args.tile_overlap
    noise_map = noisestd.expand((1, 1, tile_sz, tile_sz))

    # prepare
    os.makedirs(args.output_dir, exist_ok=True)

    # read input
    fns = [fn for fn in os.listdir(args.input_dir) if fn.endswith('.png')]
    # fns = fns[150:]
    num_frames = len(fns)

    data_list = {}
    idx = 0
    for fn in tqdm(fns):
        img = cv2.imread(os.path.join(args.input_dir, fn))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data_list[idx] = {
            'fn': fn,
            'img': tvf.to_tensor(img),
        }
        
        if idx > 1:
            inference_frame_by_idx(data_list, idx - NUM_HALF_EXT, model, noise_map, tile_sz, tile_overlap)

        idx += 1

    # last frames
    for idx in range(num_frames - NUM_HALF_EXT, num_frames):
        inference_frame_by_idx(data_list, idx, model, noise_map, tile_sz, tile_overlap)


    pass

if __name__ == "__main__":
    args = parse_args()
    main(args)


