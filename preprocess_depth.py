import numpy as np
from numpy.polynomial import Polynomial
from PIL import Image
from co3d.dataset.data_types import load_dataclass_jgzip, FrameAnnotation
from pathlib import Path
from transformers import pipeline
from tqdm import tqdm

from typing import List


def _load_16big_png_image(depth_png_path: str | Path):
    with Image.open(depth_png_path) as depth_pil:
        depth = np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
        depth = depth.astype(np.float32)
        depth = depth.reshape(*depth_pil.size[::-1])

    return depth

def _save_16big_png_image(depth_png_path: str | Path, depth: np.ndarray):
    shape = depth.shape
    depth = np.frombuffer(depth.astype(np.float16), dtype=np.uint16)
    depth = depth.reshape(*shape)

    Image.fromarray(depth).save(depth_png_path)

def _disp_to_depth(disp: np.ndarray):
    range = np.minimum(disp.max() / (disp.min() + 1e-3), 100.)
    max = disp.max()
    min = max / range

    depth = 1 / np.maximum(disp, min)
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth = np.power(depth, 1. / 2.2) # gamma

    return depth

if __name__ == "__main__":
    root = Path("./data/")
    sequences = []
    depth_estimator = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf", device='cuda:0')

    for item in root.iterdir():
        if not item.is_dir():
            continue

        annotations_path = item.joinpath('frame_annotations.jgz')

        if not annotations_path.exists():
            continue

        frame_annotations = load_dataclass_jgzip(annotations_path, List[FrameAnnotation])

        for subdir in item.iterdir():
            if not subdir.is_dir():
                continue

            if 'pointcloud' in map(lambda x: x.stem, subdir.iterdir()):
                annotations = list(filter(lambda frame: frame.sequence_name == subdir.stem, frame_annotations))
                annotations.sort(key=lambda frame: frame.frame_number)

                sequences.append(annotations)

    annotations = [annotation for sequence in sequences for annotation in sequence]

    for annotation in tqdm(annotations):
        path = root.joinpath(annotation.depth.path)
        processed_path = path.parents[1].joinpath('processed')
        processed_path.mkdir(exist_ok=True)

        depth_gt = _load_16big_png_image(path)
        rgb = Image.open(root.joinpath(annotation.image.path))
        disp = depth_estimator(rgb)['depth']
        disp = np.array(disp)

        # Do fitting only where GT depth exists.
        depth = _disp_to_depth(disp)
        masked_depth = depth[depth_gt > 0].ravel()

        if len(masked_depth) > 0:
            fit = Polynomial.fit(masked_depth, depth_gt[depth_gt > 0].ravel(), deg=1)
            fitted_depth = fit(depth.ravel()).reshape(depth.shape)
            _save_16big_png_image(processed_path.joinpath(path.name), fitted_depth)