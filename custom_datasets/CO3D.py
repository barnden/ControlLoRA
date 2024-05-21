import torch.utils.data as data
import numpy as np
from PIL import Image
import torch as th
from co3d.dataset.data_types import load_dataclass_jgzip, FrameAnnotation
from pathlib import Path
from typing import List
from posewarp import Warper
import os
from torchvision import transforms
from torchvision.transforms import functional as TF
from warpback import RGBDRenderer
from diffusers import StableDiffusionInpaintPipeline

class WarpCO3DDataset(data.Dataset):
    def __init__(self, root="./data", prompt="a photo of [V]", rare_token="xxxsy5zt", offset=1, use_depth_estimation=False, device='cuda:0'):
        if not isinstance(root, Path):
            root = Path(root)

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]
        )
        self.rare_token = rare_token
        self.prompt = prompt.replace("[V]", rare_token)
        self.root = root
        self.offset = offset
        self.use_depth_estimation = use_depth_estimation

        # For now, just use off-the-shelf SD1.5 inpaint
        # A fine-tuned model should probably be used in the future
        self.inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            'runwayml/stable-diffusion-inpainting',
            torch_dtype=th.float16,
            safety_checker=None # For some reason warps with lots of small lines likes to trigger safety trigger
        ).to('cuda:0')

        self.inpaint_pipeline.set_progress_bar_config(leave=False)
        self.inpaint_pipeline.set_progress_bar_config(disable=True)

        if use_depth_estimation:
            from transformers import pipeline
            self.depth_estimator = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf", device="cuda:0")
            self.renderer = RGBDRenderer(device)
            self.device = device
        else:
            self.warper = Warper()

        num_total_images = 0
        sequences = []
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

                    num_total_images += len(annotations) - 1

        self.sequences = sequences
        self.size = num_total_images

    def co3d_annotation_to_opencv_pose(self, entry: FrameAnnotation):
        p = entry.viewpoint.principal_point
        f = entry.viewpoint.focal_length
        h, w = entry.image.size
        K = np.eye(3)
        s = min(w,h)
        K[0, 0] = f[0] * s / 2
        K[1, 1] = f[1] * s / 2
        K[0, 2] = w / 2 - p[0] * s / 2
        K[1, 2] = h / 2 - p[1] * s / 2

        R = np.asarray(entry.viewpoint.R).T
        T = np.asarray(entry.viewpoint.T)
        pose = np.concatenate([R, T[:,None]], 1)
        pose = np.diag([-1,-1,1]).astype(np.float32) @ pose

        return K, pose

    def _load_16big_png_image(self, depth_png_path: str | Path, crop=None):
        with Image.open(depth_png_path) as depth_pil:
            if crop is not None:
                depth_pil = TF.crop(depth_pil, *crop)
            depth = np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
            depth = depth.astype(np.float32)
            depth = depth.reshape(*depth_pil.size[::-1])

        return depth

    def _depth_to_colormap(self, depth_arr: np.ndarray, cmap='inferno'):
        import matplotlib

        heatmap = depth_arr.astype(np.float32)
        heatmap /= heatmap.max()
        heatmap = matplotlib.colormaps[cmap](heatmap)
        heatmap = (heatmap * 255).astype(np.uint8)

        return Image.fromarray(heatmap)

    def __getitem__(self, index):
        if index >= self.size:
            raise ValueError("out of bounds")

        seen = 0
        for annotations in self.sequences:
            if seen + len(annotations) - 2 >= index:
                break

            seen += len(annotations) - 1

        frame = annotations[index - seen]
        next_frame = annotations[index - seen + self.offset]

        rgb = Image.open(self.root.joinpath(frame.image.path))
        rgb_copy = rgb.copy()

        maybe_resize = transforms.Resize(size=512) if rgb.size[0] < 512 or rgb.size[1] < 512 else lambda x: x
        crop_params = transforms.RandomCrop.get_params(maybe_resize(rgb), (512, 512))

        rgb = TF.crop(rgb, *crop_params)

        target = Image.open(self.root.joinpath(next_frame.image.path))
        target = TF.crop(target, *crop_params)

        K, Rt = self.co3d_annotation_to_opencv_pose(frame)
        K_tar, Rt_tar = self.co3d_annotation_to_opencv_pose(next_frame)

        # Pad [R|t] 4x4 matrix
        Rt = np.vstack((Rt, [0, 0, 0, 1]))
        Rt_tar = np.vstack((Rt_tar, [0, 0, 0, 1]))

        if 'DEBUG_CO3D' in os.environ:
            rgb.save("_debug.png")
            target.save("_debug.png")

        if self.use_depth_estimation:
            # This just uses AdaMPI's warpback module to warp from predicted disparity
            # There is no fitting between GT (if it exists) and recovered depth
            # To be used with real data

            disp = self.depth_estimator(rgb)['depth']
            disp = transforms.ToTensor()(disp)[None].to(device=self.device, dtype=th.float32)
            rgb = transforms.ToTensor()(rgb)[None].to(device=self.device, dtype=th.float32)

            Rt = th.from_numpy(Rt)[None].to(device=self.device, dtype=th.float32)
            Rt_tar = th.from_numpy(Rt_tar)[None].to(device=self.device, dtype=th.float32)

            K = th.tensor([
                [0.58, 0, 0.5],
                [0, 0.58, 0.5],
                [0, 0, 1]
            ])[None].to(self.device)

            rgbd = th.cat([rgb, disp], dim=1)

            mesh = self.renderer.construct_mesh(rgbd, K)

            Rt_render = (Rt_tar @ th.linalg.inv(Rt))[:, :3]
            rgb_warped, _, mask = self.renderer.render_mesh(mesh, K, Rt_render)
            rgb_warped = th.clamp(rgb_warped, 0., 1.)

            rgb_warped = rgb_warped.cpu().permute(0, 2, 3, 1).squeeze(0).numpy().astype(np.float32)
            inpaint_mask = (1 - mask.cpu().repeat(1, 3, 1, 1).permute(0, 2, 3, 1)).squeeze(0).numpy().astype(np.uint8)

            if 'DEBUG_CO3D' in os.environ:
                Image.fromarray((rgb_warped * 255).astype(np.uint8)).save('_debug_warp.png')
                Image.fromarray(inpaint_mask * 255).save('_debug_mask.png')
        else:
            from PIL import ImageFilter
            # Otherwise, use preprocessed predicted depth (adjusted to fit against GT depth)
            rgb = np.array(rgb)

            depth_path = self.root.joinpath(frame.depth.path)
            depth_path = depth_path.parents[1].joinpath('processed', Path(frame.depth.path).name)
            depth = self._load_16big_png_image(depth_path, crop_params)

            rgb_warped, mask, *_ = self.warper.forward_warp(rgb, None, depth, Rt, Rt_tar, K, K_tar)

            # Hacky way to dilate warped edges
            inpaint_mask = (1 - mask[..., np.newaxis].repeat(3, axis=-1).astype(np.uint8))
            inpaint_mask = inpaint_mask.clip(0, 1)
            inpaint_mask = Image.fromarray(255 * inpaint_mask).filter(ImageFilter.MaxFilter(3))
            inpaint_mask = (np.array(inpaint_mask) / 255).astype(np.uint8)

            # Apply dilated mask
            rgb_warped[inpaint_mask > 0.5] = 0

            if 'DEBUG_CO3D' in os.environ:
                self._depth_to_colormap(depth).save("_debug_depth.png")
                Image.fromarray(inpaint_mask * 255).save('_debug_mask.png')
                Image.fromarray(rgb_warped).save('_debug_warp.png')

            rgb_warped = rgb_warped.astype(np.float32) / 255.

        inpaint_mask = Image.fromarray(inpaint_mask * 255)
        rgb_warped = Image.fromarray((rgb_warped * 255).astype(np.uint8))

        inpainted_image = self.inpaint_pipeline(
            prompt="photograph, high resolution, high quality",
            negative_prompt="cartoon",
            image=rgb_warped,
            mask_image=inpaint_mask,
            strength=1.0,
            num_inference_steps=25,
            guidance_scale=1.5
        ).images[0]

        if 'DEBUG_CO3D' in os.environ:
            inpainted_image.save('_debug_inpaint.png')

        return {
            'target': self.transform(target),
            'condition': self.transform(inpainted_image),
            'prompt': self.prompt,
            'pil_original': rgb_copy,
            'pil_warp': rgb_warped,
            'pil_mask': inpaint_mask,
            'pil_target': target,
            'pil_condition': inpainted_image
        }
    
    def __len__(self):
        return self.size

if __name__ == "__main__":
    dataset = WarpCO3DDataset()

    data = dataset[4812]