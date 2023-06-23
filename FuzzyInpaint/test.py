# Copyright (c) 2022 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This repository was forked from https://github.com/openai/guided-diffusion, which is under the MIT license

"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""
import sys
sys.path.append("./FuzzyInpaint")
import os
import argparse
import torch as th
import torch.nn.functional as F
import conf_mgt
from utils import yamlread
from guided_diffusion import dist_util
from diffusers import UNet2DModel
from diffusers import DDPMScheduler
import numpy as np
from PIL import Image
import random as rand
import lovely_tensors as lt
lt.monkey_patch()


# Workaround
try:
    import ctypes
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')
except:
    pass


from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    select_args,
)  # noqa: E402



def get_var_mask(shape,min_p=1,max_p=2,width_mean=64,width_var=32):
    ''' returns a list of variable length, each element contains the x/y limit pairs of masks '''
    
    total_patches = rand.randint(min_p,max_p)
    areas = []
    
    for _ in range(total_patches):
        valid_patch = False
        while not valid_patch:
            x, y = rand.randint(0,shape[0]), rand.randint(0,shape[1])
            stretch_x, stretch_y = np.random.normal(width_mean, width_var), np.random.normal(width_mean, width_var)
            stretch_x, stretch_y = abs(round(stretch_x)), abs(round(stretch_y))
            
            if stretch_x>8 and stretch_y>8:
                if (x+stretch_x)<shape[0] and (y+stretch_y)<shape[1]:
                    valid_patch = True
                    areas.append([x,x+stretch_x,y,y+stretch_y])
    return areas
                    

def self_degrade(img):
    areas = get_var_mask(shape=img.shape[2:])
    for area in areas:
        #randomly choose which channels to degrade
        degraded_channels = []
        for channel in [0,1,2]:
            if rand.randint(0,1)>0:
                degraded_channels.append(channel)
        
        #can move val in/out the channel loop.
        degradation_val = 0 #rand.randint(-1,0)
        for c in degraded_channels:
            img[:,c,area[0]:area[1],area[2]:area[3]] = degradation_val
    return img


def toU8(sample):
    if sample is None:
        return sample

    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    sample = sample.detach().cpu().numpy()
    return sample


def imwrite(img, path, name):
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, name)
    if len(img.shape)<4:
        img = img.unsqueeze(0)
    img = toU8(img)
    Image.fromarray(img[0]).save(full_path+".png")
    return


def get_projection_zscore_mask(img, args, img_name):
    """ creates a mask based on z-score across a set of diffusion projection depths """
    img_name = img_name[0].split('.')[0]
    # Set up diffusion:
    device = img.device
    repo_id = "google/ddpm-celebahq-256"
    model = UNet2DModel.from_pretrained(repo_id)
    scheduler = DDPMScheduler.from_config(repo_id)
    model.to(device)

    th.manual_seed(0)
    T = scheduler.num_train_timesteps
    projection_set = args.get("zscore_set")

    clean_img = img
    eps = th.randn(1, model.config.in_channels, model.config.sample_size, model.config.sample_size, device=device)

    err_final = 0
    for back_t in projection_set:
        alpha_cumprod = scheduler.alphas_cumprod[back_t]
        noisy_img = np.sqrt(alpha_cumprod) * clean_img + np.sqrt(1-alpha_cumprod)*eps

        for i, t in enumerate(scheduler.timesteps[T-back_t:]):
            with th.no_grad():
                residual = model(noisy_img, t).sample
            noisy_img = scheduler.step(residual, t, noisy_img).prev_sample

        # Each intermediate gets saved with the same img name in different paths.
        name = f"{img_name}_{back_t:04d}"

        # SAVE projections.
        path = os.path.join(args["out_path"], "projected")
        imwrite(noisy_img, path, name)

        # SAVE error. {the error is <possibly> in the range [0,255**2]}
        error = th.square((noisy_img+1)*127.5 - (clean_img+1)*127.5)
        path = os.path.join(args["out_path"], "err")
        imwrite(error/(255**2)*2-1, path, name)
        print(f"t={back_t:03d}; error: min={error.min():.3f}, max={error.max():.3f}, mean={error.mean():.3f}, ")

        # Load corresponding z-score maps.
        mean = th.load(os.path.join(args["zscore_path"], f"{args['zscore_dataset']}_mean_N{args['zscore_N']}_d{back_t}"))
        std = th.load(os.path.join(args["zscore_path"], f"{args['zscore_dataset']}_std_N{args['zscore_N']}_d{back_t}"))

        # SAVE error z-score. {the zscore error should be in a smaller range, but in theory infinite range}
        # we transform the z-score: 
        #           below lower_bound is fully acceptable -> 0
        #           over upper_bound is fully unacceptable -> 1
        #           everything in between is spread [l,u] -> [0,u-l]
        err_z = (error - mean) / std
        upper_bound = args["upper_bound"] #set to full hallucination above: upper_bound(in std units)
        lower_bound = args["lower_bound"] #set to zero inpainting if under: lower_bound(in std units)
        # import pdb; pdb.set_trace()
        err_z = ( err_z.clamp(lower_bound, upper_bound) - lower_bound ) / (upper_bound-lower_bound)
        path = os.path.join(args["out_path"], "errzscore")
        imwrite( err_z *2 -1, path, name)

        err_final += err_z / len(projection_set)

        print(f"t={back_t:03d}; err_z: min={err_z.min():.3f}, max={err_z.max():.3f}, mean={err_z.mean():.3f}, ")

    print(f"t={back_t:03d}; err_final: min={err_final.min():.3f}, max={err_final.max():.3f}, mean={err_final.mean():.3f}, ")

    err_final = 1 - err_final
    path = os.path.join(args["out_path"], "mask_nolambda")
    imwrite(err_final*2-1, path, img_name)

    return err_final

def fuzzy_inpaint(conf: conf_mgt.Default_Conf, args=""):

    print("Starting", conf['name'])

    device = dist_util.dev(conf.get('device'))

    model, diffusion = create_model_and_diffusion(**select_args(conf, model_and_diffusion_defaults().keys()), conf=conf)
    model.load_state_dict(dist_util.load_state_dict(os.path.expanduser(conf.model_path), map_location="cpu"))
    model.to(device)
    if conf.use_fp16:
        model.convert_to_fp16()
    model.eval()

    show_progress = conf.show_progress

    if conf.classifier_scale > 0 and conf.classifier_path:
        print("loading classifier...")
        classifier = create_classifier(**select_args(conf, classifier_defaults().keys()))
        classifier.load_state_dict(
            dist_util.load_state_dict(os.path.expanduser(
                conf.classifier_path), map_location="cpu"))

        classifier.to(device)
        if conf.classifier_use_fp16:
            classifier.convert_to_fp16()
        classifier.eval()

        def cond_fn(x, t, y=None, gt=None, **kwargs):
            assert y is not None
            with th.enable_grad():
                x_in = x.detach().requires_grad_(True)
                logits = classifier(x_in, t)
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs[range(len(logits)), y.view(-1)]
                return th.autograd.grad(selected.sum(), x_in)[0] * conf.classifier_scale
    else:
        cond_fn = None

    def model_fn(x, t, y=None, gt=None, **kwargs):
        assert y is not None
        return model(x, t, y if conf.class_cond else None, gt=gt)

    print("sampling...")
    all_images = []

    dset = 'eval'

    eval_name = conf.get_default_eval_name()

    dl = conf.get_dataloader(dset=dset, dsName=eval_name)

    for batch in iter(dl):

        for k in batch.keys():
            if isinstance(batch[k], th.Tensor):
                batch[k] = batch[k].to(device)

        model_kwargs = {}

        model_kwargs["gt"] = batch['GT']
        
        gt_keep_mask = batch.get('gt_keep_mask')
        if gt_keep_mask is not None:
            if args:
                if args["mask_override"]=="":
                    model_kwargs['gt_keep_mask'] = gt_keep_mask
                elif args["mask_override"]=="modulate":
                    model_kwargs['gt_keep_mask'] = gt_keep_mask*args["mask_s"]
                elif args["mask_override"]=="replace":
                    model_kwargs['gt_keep_mask'] = gt_keep_mask*0+args["mask_s"]
                elif args["mask_override"]=="get_zscore":
                    if args["self_degrade"]:
                        # model_kwargs["gt"][:,:,140:220,140:220] = 0
                        model_kwargs["gt"] = self_degrade(model_kwargs["gt"])
                    model_kwargs['gt_keep_mask'] = get_projection_zscore_mask(model_kwargs["gt"], args, batch['GT_name'])
                    model_kwargs['gt_keep_mask'] = ((model_kwargs['gt_keep_mask'])**args["ood_expon"]) * args["ood_lambda"]
                else:
                    raise NotImplementedError("Choose diff mask overriding: [None, modulate, replace, get_zscore]")
            else:
                model_kwargs['gt_keep_mask'] = gt_keep_mask



        batch_size = model_kwargs["gt"].shape[0]

        if conf.cond_y is not None:
            classes = th.ones(batch_size, dtype=th.long, device=device)
            model_kwargs["y"] = classes * conf.cond_y
        else:
            classes = th.randint(low=0, high=NUM_CLASSES, size=(batch_size,), device=device)
            model_kwargs["y"] = classes

        sample_fn = diffusion.p_sample_loop if not conf.use_ddim else diffusion.ddim_sample_loop


        result = sample_fn(model_fn, (batch_size, 3, conf.image_size, conf.image_size), clip_denoised=conf.clip_denoised, 
            model_kwargs=model_kwargs, cond_fn=cond_fn, device=device, progress=show_progress, return_all=True, conf=conf)
        srs = toU8(result['sample'])
        gts = toU8(result['gt'])
        # lrs = toU8(result.get('gt') * model_kwargs.get('gt_keep_mask') + (-1) * th.ones_like(result.get('gt')) * (1 - model_kwargs.get('gt_keep_mask')))
        lrs = toU8(result['gt'] * model_kwargs['gt_keep_mask'] -1 )

        gt_keep_masks = toU8((model_kwargs.get('gt_keep_mask') * 2 - 1))

        conf.eval_imswrite(
            srs=srs, gts=gts, lrs=lrs, gt_keep_masks=gt_keep_masks,
            img_names=batch['GT_name'], dset=dset, name=eval_name, verify_same=False)

    print("sampling complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, required=False, default=None)
    args = vars(parser.parse_args())

    conf_arg = conf_mgt.conf_base.Default_Conf()
    os.chdir('./FuzzyInpaint/')

    conf = yamlread(args.get('conf_path'))
    conf_arg.update(conf)
    fuzzy_inpaint(conf_arg)
