from FuzzyInpaint.test import *
import json


def path_mod(conf, args):
    """path changer/creator for outer wrapper over the base conf"""
    conf["data"]["eval"]["paper_face_mask"]["gt_path"] = args["gt_path"]
    if args.get('mask_override')=="get_zscore" or args.get('mask_override')=="replace":
        args['mask_path'] = args["gt_path"]
    conf["data"]["eval"]["paper_face_mask"]["mask_path"] = args["mask_path"]
    conf["data"]["eval"]["paper_face_mask"]["paths"]["srs"] = os.path.join(args["out_path"], "inpainted")
    conf["data"]["eval"]["paper_face_mask"]["paths"]["lrs"] = os.path.join(args["out_path"], "gt_masked")
    conf["data"]["eval"]["paper_face_mask"]["paths"]["gts"] = os.path.join(args["out_path"], "gt_image")
    conf["data"]["eval"]["paper_face_mask"]["paths"]["gt_keep_masks"] = os.path.join(args["out_path"], "mask_used")
    return conf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    ''' Note: paths are relative to FuzzyInpaint dir

    Put images and their masks in 2 directories, as png with matching names
    Or ignore mask_path when:
            using zscore mask (it gets auto-created)
            using the replace feature that gives a uniform constant mask
    Refer to README for a walkthrough of the different features
    '''

    # DATA: Handling the in/out paths for test images
    parser.add_argument('--gt_path', type=str, required=False, default=None, help="Path to the test input images")
    parser.add_argument('--mask_path', type=str, required=False, default=None, help="! Path to masks (can be ignored by mask_override) !")
    parser.add_argument('--out_path', type=str, required=False, default="../log/Aexp", help="Path to save image outputs")

    # CONFIG: FuzzyInpaint fuzzy-conditional-diffusion
    parser.add_argument('--conf_path', type=str, required=False, default="fuzzyconfig.yml")
    parser.add_argument('--mask_override', type=str, required=False, default="", 
            help="[] read masks from mask_path, equivalent to modulate + mask_s==1 \
                  [modulate] multiply mask with scalar=mask_s, mask that was read \
                [replace] replace mask with scalar=mask_s \ [get_zscore] generate zscore mask to use")
    parser.add_argument('--mask_s', type=float, required=False, default=1.0, help="See mask_override")

    # CONFIG: Automatic mask generation with zscore method
    parser.add_argument('--zscore_path', type=str, required=False, default="../data/zscores", help="Path to the zscore tensors")
    parser.add_argument('--zscore_dataset', type=str, required=False, default="celeba", help="Name of the prior dataset")
    parser.add_argument('--zscore_N', type=int, required=False, default=1000, help="Total number of images used for the zscore")
    parser.add_argument('--zscore_set', nargs="+", required=False, default=(300,400), help="All projection depths")
    parser.add_argument('--ood_lambda', type=float, required=False, default=0.1, help="scalar that changes zscore mask strength")
    parser.add_argument('--ood_expon', type=float, required=False, default=2, help="scalar that changes zscore mask strength")
    parser.add_argument('--lower_bound', type=float, required=False, default=1., help="below lower_bound*std in MSE all is preserved from GT")
    parser.add_argument('--upper_bound', type=float, required=False, default=6., help="above upper_bound*std in MSE all is hallucinated")

    # DEGRADE: (only available in mas_override==get_zscore mode) self-degrade input images for testing
    parser.add_argument('--self_degrade', type=bool, required=False, default=False, help="if True, randomly degrades input images")
    parser.add_argument('--fuzzy_off', type=bool, required=False, default=False, help="if True, switches OFF fuzzy diffusion adaptation")


    args = vars(parser.parse_args())
    conf_arg = conf_mgt.conf_base.Default_Conf()
    os.chdir('./FuzzyInpaint/')
    conf = yamlread(args.get('conf_path'))
    conf_arg.update(conf)
    conf = path_mod(conf, args)
    conf_arg["fuzzy_off"] = args["fuzzy_off"]

    os.makedirs(args.get("out_path"), exist_ok=True)
    with open(os.path.join(args.get("out_path"), 'args.json'), 'w', encoding='utf-8') as f:
        json.dump(args, f, ensure_ascii=False, indent=4)

    fuzzy_inpaint(conf_arg, args)
