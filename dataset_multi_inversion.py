from diffusers import UNet2DModel
from diffusers import DDPMScheduler
import torch
from PIL import Image
import numpy as np
import tqdm
import os
import argparse
import glob
import torchvision.transforms as transforms
import time


def imread(path):
    pil_image = Image.open(path)
    pil_image.load()
    pil_image = pil_image.convert("RGB")
    return pil_image

def imwrite(img, path, name):
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, name)
    if len(img.shape)<4:
        img = img.unsqueeze(0)
    img = toU8(img)
    Image.fromarray(img[0]).save(full_path+".png")
    return

def toU8(sample):
    if sample is None:
        return 
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    sample = sample.detach().cpu().numpy()
    return sample



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', type=str, required=False, default="data/celeba_hq_256", help="Path to the input images")
    parser.add_argument('--out_path', type=str, required=False, default="log/celeba", help="Path to save all inverted images")
    parser.add_argument('--device', type=str, required=False, default="cuda", help="cuda for gpu comp")
    args = parser.parse_args()
    device = args.device


    # Set up diffusion:
    repo_id = "google/ddpm-celebahq-256"
    model = UNet2DModel.from_pretrained(repo_id)
    scheduler = DDPMScheduler.from_config(repo_id)
    model.to(device)

    torch.manual_seed(0)
    T = scheduler.num_train_timesteps
    projection_set =  [50,100,150,200,250,300,400,500,600,700]
    print(f"projection_set used: {projection_set}")

    # Load and process data:
    dataset = sorted(glob.glob(os.path.join(args.gt_path,'*')))
    for img_path in dataset[2000:]:
        tic = time.perf_counter()
        img_PIL = imread(img_path)
        clean_img = transforms.Compose([transforms.PILToTensor()])(img_PIL).to(device)
        clean_img = clean_img/127.5-1
        name = img_path.split(os.sep)[-1].split('.')[0]
        print(f"---  processing image # {int(name)} / {len(dataset)}  ---")
        imwrite(clean_img, args.out_path, name)

        eps = torch.randn(1, model.config.in_channels, model.config.sample_size, model.config.sample_size, device="cuda")
        for back_t in tqdm.tqdm(projection_set):
            alpha_cumprod = scheduler.alphas_cumprod[back_t]
            noisy_img = np.sqrt(alpha_cumprod) * clean_img + np.sqrt(1-alpha_cumprod)*eps

            for i, t in enumerate(scheduler.timesteps[T-back_t:]):
                with torch.no_grad():
                    residual = model(noisy_img, t).sample
                noisy_img = scheduler.step(residual, t, noisy_img).prev_sample

            imwrite(noisy_img, args.out_path, name + f"_inv{back_t:04d}")
            error = torch.square(noisy_img - clean_img)
            imwrite(error, args.out_path, name + f"_err{back_t:04d}")
        toc = time.perf_counter()
        print(f"     -> processed image {int(name)} in {(toc-tic)/60.:0.2f} min")
