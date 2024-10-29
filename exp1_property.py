import csv
import torch
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, SubsetRandomSampler

plt.style.use("seaborn-v0_8")
from ad.models import *
from fixer import *
from mydatasets import *


MVTEC_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

VISA_CATEGORIES = ['candle', 
                   'capsules', 
                   'cashew', 
                   'chewinggum', 
                   'fryum', 
                   'macaroni1', 
                   'macaroni2', 
                   'pcb1', 
                   'pcb2',
                   'pcb3', 
                   'pcb4', 
                   'pipe_fryum']

def sample(dataloader, num_samples=50):
    dataset = dataloader.dataset
    indices = np.random.permutation(len(dataset))[:num_samples]
    sampler = SubsetRandomSampler(indices)

    # Recreate the DataLoader with the new sampler
    new_dataloader = DataLoader(dataset, batch_size=1, sampler=sampler)

    return new_dataloader

def detach(ad_out):
    ad_out.score = ad_out.score.detach().cpu()
    ad_out.alpha = ad_out.alpha.detach().cpu()
    ad_out.others = {}
    # ad_out.others = detach_and_move(ad_out.others)
    return ad_out

def detach_and_move(dict_tensors):
    
    for key, value in dict_tensors.items():
        if value is not None and isinstance(value, torch.Tensor):
            dict_tensors[key] = value.detach().cpu()
            # dict_tensors[key] = [tensor.detach().cpu() for tensor in value if tensor.requires_grad]
        
    return dict_tensors

def property_metrics(x_fix, x_fix_ad_out, x_bad, x_bad_ad_out, good_parts, anom_parts):
    prop1_loss = x_fix_ad_out.score.mean()
    prop3_loss = (anom_parts * x_fix_ad_out.alpha - anom_parts * x_bad_ad_out.alpha).mean()
    prop4_loss = (good_parts * x_fix_ad_out.alpha - good_parts * x_bad_ad_out.alpha).mean()
    if len(good_parts.shape) < len(x_fix.shape):
        good_parts = good_parts[:,:,None]
        anom_parts = anom_parts[:,:,None]
    prop2_loss = F.mse_loss(good_parts * x_fix, good_parts * x_bad, reduction="sum").sqrt()
    return torch.tensor([prop1_loss, prop2_loss, prop3_loss, prop4_loss])

def save_plot(x, fp):
    x = x.detach().cpu()
    batch = x.size(0)

    for i in range(batch):
        # Create a new figure for each image to ensure they don't overlap
        fig, ax = plt.subplots()

        # Display the image
        ax.imshow(x[i].numpy().transpose(1, 2, 0))
        ax.axis('off')  # Hide the axes
        
        # Set the background color of the figure to none (transparent)
        fig.patch.set_facecolor('none')
        fig.patch.set_edgecolor('none')

        # Modify the file path to include the index for each image
        # Assuming fp includes the file extension, we insert the index before the extension
        file_extension = fp.split('.')[-1]
        new_fp = f"{fp[:-len(file_extension)-1]}_{i}.{file_extension}"

        # Save the current figure with a transparent background
        plt.savefig(new_fp, transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close(fig) 
        
def plot_timeseries(batch, ys, x_bad, x_fix, x_fix_baseline, index=2, model_name="gpt2"):
    os.makedirs(f"_dump/{model_name}/swat/", exist_ok=True)
    rows_with_ones = torch.any(ys == 1, dim=1)
    row = torch.nonzero(rows_with_ones, as_tuple=True)[0][-1]
    list1 = x_bad.detach().cpu()[row, :,index]
    list2 = x_fix.detach().cpu()[row, :, index]
    list3 = x_fix_baseline.detach().cpu()[row, :, index]
    
    xs = list(range(len(list1)))

    # Plotting
    plt.figure(figsize=(10, 6))

    # Base bar plot with light grey color
    # plt.bar(xs, list1, color='lightgreen', width=1, alpha=0.3)

    # Overlay red color on anomalies
    colors = ['mistyrose' if value == 1.0 else 'none' for value in ys[row]]
    plt.bar(xs, list1, color=colors, width=1)

    # Plotting lines
    plt.plot(list1.numpy(), label='original', linestyle='--')
    plt.plot(list2.numpy(), label='guided', linestyle='-')
    plt.plot(list3.numpy(), label='baseline', linestyle='-.')

    plt.xlabel("Timestamps", fontsize=20)
    plt.ylabel("Sensor Reading", fontsize=20)
    plt.tick_params(axis='both', labelsize=20)
    plt.legend(fontsize=16)
    plt.show()
    plt.savefig(f"_dump/{model_name}/swat/batch_{batch}_row_{row}_index_{index}.png")
    plt.close()

def save_csv(i, metrics, fp):
    file = open(fp, 'a')
    writer = csv.writer(file)
    metrics = metrics.tolist()
    metrics.insert(0, i)
    writer.writerow(metrics)  # Write headers
    file.close()

def eval_image_property_improvement(
        dataset, 
        category, 
        image_size=512, 
        batch_size=1, 
        quantile=0.9, 
        noise_level=1000,
        prop1_scale=0.1,
        prop2_scale=0.1,
        prop3_scale=1.0,
        prop4_scale=1.0,
        guide_scale=0.1):
    # load ad model
    ad = FastflowAdModel(image_size=image_size)
    state_dict = torch.load(f"_dump/ad_noisy_fast_wide_resnet50_2_{dataset}_{category}_{image_size}_best.pt")["model_state_dict"]
    ad.load_state_dict(state_dict)
    ad.eval().cuda();
    # load dataloader
    dataloader = get_fixer_dataloader(dataset, batch_size=batch_size, category=category, split="test", image_size=image_size)

    # load fixer model
    model_path = f"_dump/fixer_diffusion_{dataset}_{category}_best.pt"
    model_dict = torch.load(model_path)['model_state_dict']
    mydiff = MyDiffusionModel(image_size=image_size)
    mydiff.load_state_dict(model_dict)
    mydiff.eval().cuda();

    # repair config
    repair_config = VisionRepairConfig(category=category, 
                                       lr=1e-5, 
                                        batch_size=batch_size, 
                                        prop1_scale=prop1_scale, 
                                        prop2_scale=prop2_scale,
                                        prop3_scale=prop3_scale,
                                        prop4_scale=prop4_scale,
                                        guide_scale=guide_scale)
    # repair config
    baseline_infill_config = VisionRepairConfig(category=category, 
                                        lr=1e-5, 
                                        batch_size=batch_size, 
                                        prop1_scale=prop1_scale, 
                                        prop2_scale=prop2_scale,
                                        prop3_scale=prop3_scale,
                                        prop4_scale=prop4_scale,
                                        guide_scale=0.0)
    os.makedirs(f"_dump/{dataset}/{category}/", exist_ok=True)

    # dataloader = sample(dataloader, 50)
    metrics_base = []
    metrics_ours = []
    for i, batch in enumerate(tqdm(dataloader)):
        bad_idxs = (batch['label'] != 0)
        if bad_idxs.sum() < 1:
            continue
        x_bad, y, m = batch['image'][bad_idxs], batch['label'][bad_idxs], batch['mask'][bad_idxs]
        x_bad = (2*x_bad-1).cuda()
        if x_bad.size(1) == 1:
            x_bad = x_bad.repeat(1, 3, 1, 1)

        x_bad_ad_out = ad(x_bad)
        anom_parts = (x_bad_ad_out.alpha > x_bad_ad_out.alpha.view(x_bad.size(0),-1).quantile(quantile,dim=1).view(-1,1,1,1)).long()
        good_parts = 1 - anom_parts

        average_colors = (x_bad * anom_parts).sum(dim=(-1,-2)) / (anom_parts.sum(dim=(-1,-2)))
        x_bad_masked = (1-anom_parts) * x_bad + anom_parts * (average_colors.view(-1,3,1,1))
        save_plot(0.5*x_bad+0.5, f"_dump/{dataset}/{category}/{i}_x_bad.png")
        save_plot(m, f"_dump/{dataset}/{category}/{i}_x_bad_gt.png")
        save_plot(0.5*x_bad_masked+0.5, f"_dump/{dataset}/{category}/{i}_x_bad_masked.png")

        

        ## baseline infill
        infill_out = vision_repair(x_bad, anom_parts, ad, mydiff, baseline_infill_config, noise_level)
        x_fix_baseline_infill = infill_out['x_fix'].clamp(-1,1)
        save_plot(0.5*x_fix_baseline_infill+0.5, f"_dump/{dataset}/{category}/{i}_x_fix_baseline_infill.png")
        x_fix_baseline_infill_ad_out = ad(x_fix_baseline_infill)
       
        ## our method
        out = vision_repair(x_bad, anom_parts, ad, mydiff, repair_config, noise_level)
        x_fix = out['x_fix'].clamp(-1,1)
        x_fix_ad_out = ad(x_fix)
        save_plot(0.5*x_fix+0.5, f"_dump/{dataset}/{category}/{i}_x_fix.png")

        ## baseline SDEdit
        x_fix_baseline = mydiff(x_bad_masked.cuda(), noise_level, num_inference_steps=1000, progress_bar=True)
        x_fix_baseline = x_fix_baseline.clamp(-1,1)
        save_plot(0.5*x_fix_baseline+0.5, f"_dump/{dataset}/{category}/{i}_x_fix_baseline.png")
        x_fix_baseline_ad_out = ad(x_fix_baseline)

        x_fix_baseline = x_fix_baseline.detach().cpu()
        x_fix_baseline_infill = x_fix_baseline_infill.detach().cpu()
        x_fix_baseline_infill_ad_out = detach(x_fix_baseline_infill_ad_out)
        x_fix_baseline_ad_out = detach(x_fix_baseline_ad_out)
        x_bad = x_bad.detach().cpu()
        x_bad_ad_out = detach(x_bad_ad_out)
        x_fix = x_fix.detach().cpu()
        x_fix_ad_out = detach(x_fix_ad_out)
        anom_parts = anom_parts.detach().cpu()
        good_parts = good_parts.detach().cpu()
        metric_base = property_metrics(x_fix_baseline, x_fix_baseline_ad_out, x_bad, x_bad_ad_out, good_parts, anom_parts)
        metric_base_infill = property_metrics(x_fix_baseline_infill, x_fix_baseline_infill_ad_out, x_bad, x_bad_ad_out, good_parts, anom_parts)
        metric_ours = property_metrics(x_fix, x_fix_ad_out, x_bad, x_bad_ad_out, good_parts, anom_parts)
        save_csv(i, metric_base, f"_dump/results/{dataset}_{category}"
                                                          f"_p1_{prop1_scale}"
                                                          f"_p2_{prop2_scale}"
                                                          f"_p3_{prop3_scale}"
                                                          f"_p4_{prop4_scale}"
                                                          f"_end_{guide_scale}"
                                                          f"_baseline.csv")
        save_csv(i, metric_base_infill, f"_dump/results/{dataset}_{category}"
                                                          f"_p1_{prop1_scale}"
                                                          f"_p2_{prop2_scale}"
                                                          f"_p3_{prop3_scale}"
                                                          f"_p4_{prop4_scale}"
                                                          f"_end_{guide_scale}"
                                                          f"_baseline_infill.csv")
        save_csv(i, metric_ours, f"_dump/results/{dataset}_{category}"
                                                          f"_p1_{prop1_scale}"
                                                          f"_p2_{prop2_scale}"
                                                          f"_p3_{prop3_scale}"
                                                          f"_p4_{prop4_scale}"
                                                          f"_end_{guide_scale}"
                                                          f"_guided.csv")
        metrics_base.append(metric_base)
        metrics_ours.append(metric_ours)
       
        torch.cuda.empty_cache()

def eval_image_property_improvement_eff(
        dataset, 
        category, 
        image_size=256, 
        batch_size=1, 
        quantile=0.9, 
        noise_level=1000,
        prop1_scale=0.1,
        prop2_scale=0.1,
        prop3_scale=1.0,
        prop4_scale=1.0,
        guide_scale=0.1):
    # load ad model
    ad = EfficientAdADModel(model_size="medium")
    state_dict = torch.load(f"_dump/ad_eff_{dataset}_{category}_best.pt")["model_state_dict"]
    ad.load_state_dict(state_dict)
    ad.eval().cuda();
    # load dataloader
    dataloader = get_fixer_dataloader(dataset, batch_size=batch_size, category=category, split="test", image_size=image_size)

    # load fixer model
    model_path = f"_dump/fixer_diffusion_{dataset}_{category}_best.pt"
    model_dict = torch.load(model_path)['model_state_dict']
    mydiff = MyDiffusionModel(image_size=image_size)
    mydiff.load_state_dict(model_dict)
    mydiff.eval().cuda();

    # repair config
    repair_config = VisionRepairConfig(category=category, 
                                       lr=1e-5, 
                                        batch_size=batch_size, 
                                        prop1_scale=prop1_scale, 
                                        prop2_scale=prop2_scale,
                                        prop3_scale=prop3_scale,
                                        prop4_scale=prop4_scale,
                                        guide_scale=guide_scale)
    # repair config
    baseline_infill_config = VisionRepairConfig(category=category, 
                                        lr=1e-5, 
                                        batch_size=batch_size, 
                                        prop1_scale=prop1_scale, 
                                        prop2_scale=prop2_scale,
                                        prop3_scale=prop3_scale,
                                        prop4_scale=prop4_scale,
                                        guide_scale=0.0)
    os.makedirs(f"_dump/eff/{dataset}/{category}/", exist_ok=True)

    # dataloader = sample(dataloader, 50)
    metrics_base = []
    metrics_ours = []
    cnt = 0
    for i, batch in enumerate(tqdm(dataloader)):
        bad_idxs = (batch['label'] != 0)
        
        if bad_idxs.sum() < 1:
            continue
        x_bad, y, m = batch['image'][bad_idxs], batch['label'][bad_idxs], batch['mask'][bad_idxs]
        x_bad = (2*x_bad-1).cuda()
        if x_bad.size(1) == 1:
            x_bad = x_bad.repeat(1, 3, 1, 1)
        x_bad_ad_out = ad(x_bad)
        anom_parts = (x_bad_ad_out.alpha > x_bad_ad_out.alpha.view(x_bad.size(0),-1).quantile(quantile,dim=1).view(-1,1,1,1)).long()
        good_parts = 1 - anom_parts

        average_colors = (x_bad * anom_parts).sum(dim=(-1,-2)) / (anom_parts.sum(dim=(-1,-2)))
        x_bad_masked = (1-anom_parts) * x_bad + anom_parts * (average_colors.view(-1,3,1,1))
        save_plot(0.5*x_bad+0.5, f"_dump/eff/{dataset}/{category}/{i}_x_bad.png")
        save_plot(m, f"_dump/eff/{dataset}/{category}/{i}_x_bad_gt.png")
        save_plot(0.5*x_bad_masked+0.5, f"_dump/eff/{dataset}/{category}/{i}_x_bad_masked.png")

        

        ## baseline infill
        infill_out = vision_repair(x_bad, anom_parts, ad, mydiff, baseline_infill_config, noise_level)
        x_fix_baseline_infill = infill_out['x_fix'].clamp(-1,1)
        save_plot(0.5*x_fix_baseline_infill+0.5, f"_dump/eff/{dataset}/{category}/{i}_x_fix_baseline_infill.png")
        x_fix_baseline_infill_ad_out = ad(x_fix_baseline_infill)
       
        ## our method
        out = vision_repair(x_bad, anom_parts, ad, mydiff, repair_config, noise_level)
        x_fix = out['x_fix'].clamp(-1,1)
        x_fix_ad_out = ad(x_fix)
        save_plot(0.5*x_fix+0.5, f"_dump/eff/{dataset}/{category}/{i}_x_fix.png")

        ## baseline SDEdit
        x_fix_baseline = mydiff(x_bad_masked.cuda(), noise_level, num_inference_steps=1000, progress_bar=True)
        x_fix_baseline = x_fix_baseline.clamp(-1,1)
        save_plot(0.5*x_fix_baseline+0.5, f"_dump/eff/{dataset}/{category}/{i}_x_fix_baseline.png")
        x_fix_baseline_ad_out = ad(x_fix_baseline)

        x_fix_baseline = x_fix_baseline.detach().cpu()
        x_fix_baseline_infill = x_fix_baseline_infill.detach().cpu()
        x_fix_baseline_infill_ad_out = detach(x_fix_baseline_infill_ad_out)
        x_fix_baseline_ad_out = detach(x_fix_baseline_ad_out)
        x_bad = x_bad.detach().cpu()
        x_bad_ad_out = detach(x_bad_ad_out)
        x_fix = x_fix.detach().cpu()
        x_fix_ad_out = detach(x_fix_ad_out)
        anom_parts = anom_parts.detach().cpu()
        good_parts = good_parts.detach().cpu()
        metric_base = property_metrics(x_fix_baseline, x_fix_baseline_ad_out, x_bad, x_bad_ad_out, good_parts, anom_parts)
        metric_base_infill = property_metrics(x_fix_baseline_infill, x_fix_baseline_infill_ad_out, x_bad, x_bad_ad_out, good_parts, anom_parts)
        metric_ours = property_metrics(x_fix, x_fix_ad_out, x_bad, x_bad_ad_out, good_parts, anom_parts)
        save_csv(i, metric_base, f"_dump/results/eff_{dataset}_{category}"
                                                          f"_p1_{prop1_scale}"
                                                          f"_p2_{prop2_scale}"
                                                          f"_p3_{prop3_scale}"
                                                          f"_p4_{prop4_scale}"
                                                          f"_end_{guide_scale}"
                                                          f"_baseline.csv")
        save_csv(i, metric_base_infill, f"_dump/results/eff_{dataset}_{category}"
                                                          f"_p1_{prop1_scale}"
                                                          f"_p2_{prop2_scale}"
                                                          f"_p3_{prop3_scale}"
                                                          f"_p4_{prop4_scale}"
                                                          f"_end_{guide_scale}"
                                                          f"_baseline_infill.csv")
        save_csv(i, metric_ours, f"_dump/results/eff_{dataset}_{category}"
                                                          f"_p1_{prop1_scale}"
                                                          f"_p2_{prop2_scale}"
                                                          f"_p3_{prop3_scale}"
                                                          f"_p4_{prop4_scale}"
                                                          f"_end_{guide_scale}"
                                                          f"_guided.csv")
        metrics_base.append(metric_base)
        metrics_ours.append(metric_ours)
        cnt += 1
        if cnt == 0:
            break
       
        torch.cuda.empty_cache()
        

def get_ts_threshold(model, train_dataloader, test_dataloader, dataset="swat", model_name="gpt2"):
    attens_energy = []
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_m) in enumerate(tqdm(train_dataloader)):
            if i > 1000:
                break

            batch_x = batch_x.float().cuda()
            # reconstruction
            outputs = model(batch_x)
            # criterion
            score = torch.mean(outputs.alpha, dim=-2)
            
            score = score.detach().cpu().numpy()
            attens_energy.append(score)

    attens_energy = np.concatenate(attens_energy, axis=0)
    train_energy = np.array(attens_energy)
    attens_energy = []
    test_labels = []
    for i, batch in enumerate(tqdm(test_dataloader)):
        if i > 1000:
            break
        x, y, _  = batch
        x = x.cuda()
        outputs = model(x)
        score = torch.mean(outputs.alpha, dim=-2)
        score = score.detach().cpu().numpy()
        attens_energy.append(score)
        test_labels.append(y)
        
    attens_energy = np.concatenate(attens_energy, axis=0)
    test_energy = np.array(attens_energy)
    combined_energy = np.concatenate([train_energy, test_energy], axis=0)
    threshold = torch.tensor(np.percentile(combined_energy, 99, axis=0))
    # pred = (attens_energy > threshold).long()
    torch.save(threshold,f"_dump/{dataset}_threshold_{model_name}.pt")
    return threshold

def plot_ts(x, y, idx, fp):
    x_to_plot = x[:, idx].detach().cpu()
    fig, ax1 = plt.subplots(figsize=(8, 4))

    # Plot x on the left y-axis
    ax1.plot(x_to_plot, marker='o', linestyle='-', color='blue', label='X Values')
    ax1.set_xlabel('Window Index')
    ax1.set_ylabel('X Values', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True)

    # Highlight anomalies
    for i, label in enumerate(y):
        if label == 1:
            ax1.axvspan(i-0.5, i+0.5, color='red', alpha=0.3)

    # Adding legends
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines , labels, loc='upper left')
    plt.savefig(fp)

def eval_time_property_improvement_llama(
        dataset, 
        window_size=100, 
        batch_size=16, 
        noise_level=500, 
        prop1_scale=0.1,
        prop2_scale=0.1,
        prop3_scale=1.0,
        prop4_scale=1.0,
        guide_scale=0.1,
        num_inference_steps=200):
    model_kwargs = {
    "coef": 1e-2,
    "learning_rate": 5e-2
}
    if dataset == "wadi":
        num_features = 127
    elif dataset == "swat":
        num_features = 51
    else:
        num_features = 86
    # load fixer
    path = f"/home/antonxue/foo/arpro/_dump/fixer_ts_diffusion_{dataset}_best.pt"
    mytsdiff = MyTimeDiffusionModel(feature_dim=num_features, window_size=window_size)
    model_dict = torch.load(path)['model_state_dict']
    mytsdiff.load_state_dict(model_dict)
    mytsdiff.eval().cuda();

    # load ad
    ad = Llama2ADModel(num_features=num_features)
    path = f'/home/antonxue/foo/arpro/_dump/ad_llama2_{dataset}_best.pt'
    model_dict = torch.load(path)['model_state_dict']
    ad.load_state_dict(model_dict)
    ad.eval().cuda();

    # load dataloader
    time_ds = get_timeseries_bundle(ds_name=dataset, label_choice = 'all', shuffle=False, test_batch_size=batch_size, train_has_only_goods=True)
    train_dataloader = time_ds['train_dataloader']
    test_dataloader = time_ds['test_dataloader'] 
    time_config = TimeRepairConfig(lr=1e-5, 
                                   batch_size=batch_size, 
                                   prop1_scale=prop1_scale, 
                                   prop2_scale=prop2_scale,
                                   prop3_scale=prop3_scale,
                                   prop4_scale=prop4_scale,
                                   guide_scale=guide_scale
                                   )
    base_config =  TimeRepairConfig(lr=1e-5, 
                                   batch_size=batch_size, 
                                   prop1_scale=prop1_scale, 
                                   prop2_scale=prop2_scale,
                                   prop3_scale=prop3_scale,
                                   prop4_scale=prop4_scale,
                                   guide_scale=0.0
                                   )
    print("getting threshold")
    # threshold = get_ts_threshold(ad, train_dataloader, test_dataloader, dataset, "llama2")
    threshold = torch.load(f"_dump/{dataset}_threshold_llama2.pt")
    # value: 0.00017069593071937592

    metrics_base = []
    metrics_ours = []
    for i, batch in enumerate(tqdm(test_dataloader)):
        x, y, m  = batch
        if y.sum() == 0:
            continue
        
        x_bad = x.cuda()
        x_bad_ad_out = ad(x_bad)
        anom_parts = (x_bad_ad_out.alpha > threshold.cuda()).long()
        good_parts = (1 - anom_parts).long()

        # guided
        guided_ret = time_repair(x_bad, anom_parts, ad, mytsdiff, time_config, noise_level, num_inference_steps=num_inference_steps)
        x_fix = guided_ret['x_fix']
        x_fix_ad_out = ad(x_fix)

        # baseline
        x_fix_baseline = mytsdiff.repair(x_bad, x_bad * good_parts, good_parts, model_kwargs=model_kwargs, noise_level=noise_level, num_inference_steps=num_inference_steps)
        x_fix_baseline_ad_out = ad(x_fix_baseline)
        # # guided
        # base_ret = time_repair(x_bad, anom_parts, ad, mytsdiff, base_config, noise_level, num_inference_steps=num_inference_steps)
        # x_fix_baseline = base_ret['x_fix']
        # x_fix_baseline_ad_out = ad(x_fix_baseline)

        x_fix_baseline = x_fix_baseline.detach().cpu()
        x_fix_baseline_ad_out = detach(x_fix_baseline_ad_out)
        x_bad = x_bad.detach().cpu()
        x_bad_ad_out = detach(x_bad_ad_out)
        x_fix = x_fix.detach().cpu()
        x_fix_ad_out = detach(x_fix_ad_out)
        anom_parts = anom_parts.detach().cpu()
        good_parts = good_parts.detach().cpu()

        metric_base = property_metrics(x_fix_baseline, x_fix_baseline_ad_out, x_bad, x_bad_ad_out, good_parts, anom_parts)
        metric_ours = property_metrics(x_fix, x_fix_ad_out, x_bad, x_bad_ad_out, good_parts, anom_parts)
        save_csv(i, metric_base,   f"_dump/results/llama_{dataset}_p1_{prop1_scale}"
                                                          f"_p2_{prop2_scale}"
                                                          f"_p3_{prop3_scale}"
                                                          f"_p4_{prop4_scale}"
                                                          f"_end_{guide_scale}"
                                                          f"_steps_{num_inference_steps}"
                                                          f"_baseline.csv")
        save_csv(i, metric_ours,   f"_dump/results/llama_{dataset}_p1_{prop1_scale}"
                                                          f"_p2_{prop2_scale}"
                                                          f"_p3_{prop3_scale}"
                                                          f"_p4_{prop4_scale}"
                                                          f"_end_{guide_scale}"
                                                          f"_steps_{num_inference_steps}"
                                                          f"_guided.csv")
        metrics_base.append(metric_base)
        metrics_ours.append(metric_ours)
        torch.cuda.empty_cache()

        mask = (m == 1).any(dim=1).any(dim=0)
        anom_channels = torch.nonzero(mask, as_tuple=True)[0]
        for idx in anom_channels:
            plot_timeseries(i, y, x_bad, x_fix, x_fix_baseline, idx, model_name="llama2")

def eval_time_property_improvement(
        dataset, 
        window_size=100, 
        batch_size=16, 
        noise_level=500, 
        prop1_scale=0.1,
        prop2_scale=0.1,
        prop3_scale=1.0,
        prop4_scale=1.0,
        guide_scale=0.1,
        num_inference_steps=200):
    model_kwargs = {
    "coef": 1e-2,
    "learning_rate": 5e-2
}
    if dataset == "wadi":
        num_features = 127
    elif dataset == "swat":
        num_features = 51
    else:
        num_features = 86
    # load fixer
    path = f"/home/antonxue/foo/arpro/_dump/fixer_ts_diffusion_{dataset}_best.pt"
    mytsdiff = MyTimeDiffusionModel(feature_dim=num_features, window_size=window_size)
    model_dict = torch.load(path)['model_state_dict']
    mytsdiff.load_state_dict(model_dict)
    mytsdiff.eval().cuda();

    # load ad
    ad = GPT2ADModel(num_features=num_features)
    path = f'/home/antonxue/foo/arpro/_dump/ad_gpt2_{dataset}_best.pt'
    model_dict = torch.load(path)['model_state_dict']
    ad.load_state_dict(model_dict)
    ad.eval().cuda();

    # load dataloader
    time_ds = get_timeseries_bundle(ds_name=dataset, label_choice = 'all', shuffle=False, test_batch_size=batch_size, train_has_only_goods=True)
    train_dataloader = time_ds['train_dataloader'] 
    test_dataloader = time_ds['test_dataloader'] 
    time_config = TimeRepairConfig(lr=1e-5, 
                                   batch_size=batch_size, 
                                   prop1_scale=prop1_scale, 
                                   prop2_scale=prop2_scale,
                                   prop3_scale=prop3_scale,
                                   prop4_scale=prop4_scale,
                                   guide_scale=guide_scale
                                   )
    base_config =  TimeRepairConfig(lr=1e-5, 
                                   batch_size=batch_size, 
                                   prop1_scale=prop1_scale, 
                                   prop2_scale=prop2_scale,
                                   prop3_scale=prop3_scale,
                                   prop4_scale=prop4_scale,
                                   guide_scale=0.0
                                   )
    # threshold = get_ts_threshold(ad, train_dataloader, test_dataloader, dataset, "gpt2")
    threshold = torch.load(f"_dump/{dataset}_threshold_gpt2.pt")
    # value: 0.00017069593071937592

    metrics_base = []
    metrics_ours = []
    for i, batch in enumerate(test_dataloader):
        x, y, m  = batch
        if y.sum() == 0:
            continue
        
        x_bad = x.cuda()
        x_bad_ad_out = ad(x_bad)
        anom_parts = (x_bad_ad_out.alpha > threshold.cuda()).long()
        good_parts = (1 - anom_parts).long()

        # guided
        guided_ret = time_repair(x_bad, anom_parts, ad, mytsdiff, time_config, noise_level, num_inference_steps=num_inference_steps)
        x_fix = guided_ret['x_fix']
        x_fix_ad_out = ad(x_fix)

        # baseline
        x_fix_baseline = mytsdiff.repair(x_bad, x_bad * good_parts, good_parts, model_kwargs=model_kwargs, noise_level=noise_level, num_inference_steps=num_inference_steps)
        x_fix_baseline_ad_out = ad(x_fix_baseline)
        # # guided
        # base_ret = time_repair(x_bad, anom_parts, ad, mytsdiff, base_config, noise_level, num_inference_steps=num_inference_steps)
        # x_fix_baseline = base_ret['x_fix']
        # x_fix_baseline_ad_out = ad(x_fix_baseline)

        x_fix_baseline = x_fix_baseline.detach().cpu()
        x_fix_baseline_ad_out = detach(x_fix_baseline_ad_out)
        x_bad = x_bad.detach().cpu()
        x_bad_ad_out = detach(x_bad_ad_out)
        x_fix = x_fix.detach().cpu()
        x_fix_ad_out = detach(x_fix_ad_out)
        anom_parts = anom_parts.detach().cpu()
        good_parts = good_parts.detach().cpu()

        metric_base = property_metrics(x_fix_baseline, x_fix_baseline_ad_out, x_bad, x_bad_ad_out, good_parts, anom_parts)
        metric_ours = property_metrics(x_fix, x_fix_ad_out, x_bad, x_bad_ad_out, good_parts, anom_parts)
        save_csv(i, metric_base,   f"_dump/results/{dataset}_p1_{prop1_scale}"
                                                          f"_p2_{prop2_scale}"
                                                          f"_p3_{prop3_scale}"
                                                          f"_p4_{prop4_scale}"
                                                          f"_end_{guide_scale}"
                                                          f"_steps_{num_inference_steps}"
                                                          f"_baseline.csv")
        save_csv(i, metric_ours,   f"_dump/results/{dataset}_p1_{prop1_scale}"
                                                          f"_p2_{prop2_scale}"
                                                          f"_p3_{prop3_scale}"
                                                          f"_p4_{prop4_scale}"
                                                          f"_end_{guide_scale}"
                                                          f"_steps_{num_inference_steps}"
                                                          f"_guided.csv")
        metrics_base.append(metric_base)
        metrics_ours.append(metric_ours)
        torch.cuda.empty_cache()

        mask = (m == 1).any(dim=1).any(dim=0)
        anom_channels = torch.nonzero(mask, as_tuple=True)[0]
        for idx in anom_channels:
            plot_timeseries(i, y, x_bad, x_fix, x_fix_baseline, idx)

def save_text(original, x_fix_baseline, x_fix, anom_parts, tokenizer, fp):
    orgl = [tokenizer.decode(encoded) for encoded in original]
    base = [tokenizer.decode(encoded) for encoded in x_fix_baseline]
    guide = [tokenizer.decode(encoded) for encoded in x_fix]
    file = open(fp, 'a')
    file.write(f"original:\n {orgl[0]}\n\n")
    file.write(f"baseline:\n {base[0]}\n\n")
    file.write(f"guided:\n {guide[0]}\n\n")
    file.write(f"anom_parts:\n {anom_parts.tolist()[0]}")
    file.close()
    
        
def eval_text_property_improvement(
        dataset, 
        quantile=0.9,
        batch_size=1, 
        noise_level=900,
        num_inference_steps=1000,  
        prop1_scale=0.1,
        prop2_scale=0.1,
        prop3_scale=1.0,
        prop4_scale=1.0,
        guide_scale=0.1):
    
    # load diffusion model
    mytextdiff = MyTextDiffusionModel(num_embeddings=50265, embedding_dim=768)
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base-openai-detector").from_pretrained("roberta-base-openai-detector")
    model_path = "_dump/fixer_diffusion_webtext_best.pt"
    model_dict = torch.load(model_path)['model_state_dict']
    mytextdiff.load_state_dict(model_dict)
    mytextdiff.eval().cuda();
    dlm = mytextdiff.dlm_model

    # load roberta, rember to flip
    ad = RobertaADModel()
    ad.eval().cuda();

    # load dataloader
    # real: 0 | fake: 1
    test_dataloader = get_fixer_dataloader(
                            dataset_name = "webtext",
                            batch_size = batch_size,
                            category = None,
                            split = "test")
    repair_config = TextRepairConfig(lr=1e-5, 
                            batch_size=batch_size, 
                            prop1_scale=prop1_scale, 
                            prop2_scale=prop2_scale,
                            prop3_scale=prop3_scale,
                            prop4_scale=prop4_scale,
                            guide_scale=guide_scale)
    
    for i, batch in enumerate(test_dataloader):
        ids, length, mask, y  = batch
        if y.sum() == 0:
            continue
        x_bad = ids.cuda()
        x_bad_emb = dlm.get_embeddings(x_bad)
        x_bad_ad_out = ad(x_bad)
        anom_parts = (x_bad_ad_out.alpha > x_bad_ad_out.alpha.view(x_bad.size(0),-1).quantile(quantile,dim=1).view(x_bad.size(0),-1)).long()
        good_parts = (1 - anom_parts).long()

        # baseline
        x_fix_baseline = mytextdiff(x_bad, noise_level, num_inference_steps=num_inference_steps, progress_bar=True, batch_size=batch_size)
        x_fix_baseline_ad_out = ad(x_fix_baseline)
        x_fix_baseline_emb = dlm.get_embeddings(x_fix_baseline)
        metric_base = property_metrics(x_fix_baseline_emb, x_fix_baseline_ad_out, x_bad_emb, x_bad_ad_out, good_parts, anom_parts)

        # guided
        out = text_repair(x_bad, anom_parts, ad, mytextdiff, repair_config, noise_level, progress_bar=True, num_inference_steps=num_inference_steps)
        x_fix = out['x_fix']
        x_fix_ad_out = ad(x_fix)
        x_fix_emb = dlm.get_embeddings(x_fix)
        metric_ours = property_metrics(x_fix_emb, x_fix_ad_out, x_bad_emb, x_bad_ad_out, good_parts, anom_parts)

        save_text(x_bad, x_fix_baseline, x_fix, anom_parts, tokenizer, f"_dump/webtext/batch_{i}_y_{y.item()}.txt")

        # detach stuffs
        x_fix_baseline = x_fix_baseline.detach().cpu()
        x_fix_baseline_ad_out = detach(x_fix_baseline_ad_out)
        x_bad = x_bad.detach().cpu()
        x_bad_ad_out = detach(x_bad_ad_out)
        x_fix = x_fix.detach().cpu()
        x_fix_ad_out = detach(x_fix_ad_out)
        anom_parts = anom_parts.detach().cpu()
        good_parts = good_parts.detach().cpu()

        # save stuffs
        save_csv(i, metric_base, f"_dump/results/{dataset}"
                                            f"_p1_{prop1_scale}"
                                            f"_p2_{prop2_scale}"
                                            f"_p3_{prop3_scale}"
                                            f"_p4_{prop4_scale}"
                                            f"_end_{guide_scale}"
                                            f"_baseline.csv")
        save_csv(i, metric_ours, f"_dump/results/{dataset}"
                                            f"_p1_{prop1_scale}"
                                            f"_p2_{prop2_scale}"
                                            f"_p3_{prop3_scale}"
                                            f"_p4_{prop4_scale}"
                                            f"_end_{guide_scale}"
                                            f"_guided.csv")
        torch.cuda.empty_cache()
        

    


def compute_stats(
        model,
        dataset, 
        category=None,
        prop1_scale=1.0,
        prop2_scale=1.0,
        prop3_scale=1.0,
        prop4_scale=1.0,
        guide_scale=10.0,
        steps=500):
    columns = ['m1', 'm2', 'm3', 'm4']
    if dataset == "swat" or dataset == "wadi" or dataset == "hai":
        if model == "llama":
            base_path = (f"_dump/results/llama_{dataset}_p1_{prop1_scale}"
                                                f"_p2_{prop2_scale}"
                                                f"_p3_{prop3_scale}"
                                                f"_p4_{prop4_scale}"
                                                f"_end_{guide_scale}"
                                                f"_steps_{steps}"
                                                f"_baseline.csv")
            guided_path = (f"_dump/results/llama_{dataset}_p1_{prop1_scale}"
                                                f"_p2_{prop2_scale}"
                                                f"_p3_{prop3_scale}"
                                                f"_p4_{prop4_scale}"
                                                f"_end_{guide_scale}"
                                                f"_steps_{steps}"
                                                f"_guided.csv")
        else:
            base_path = (f"_dump/results/{dataset}_p1_{prop1_scale}"
                                                f"_p2_{prop2_scale}"
                                                f"_p3_{prop3_scale}"
                                                f"_p4_{prop4_scale}"
                                                f"_end_{guide_scale}"
                                                f"_steps_{steps}"
                                                f"_baseline.csv")
            guided_path = (f"_dump/results/{dataset}_p1_{prop1_scale}"
                                                f"_p2_{prop2_scale}"
                                                f"_p3_{prop3_scale}"
                                                f"_p4_{prop4_scale}"
                                                f"_end_{guide_scale}"
                                                f"_steps_{steps}"
                                                f"_guided.csv")
    elif dataset == "visa" or dataset == "mvtec":
        if model == "eff":
            base_path = (f"_dump/results/eff_{dataset}_{category}"
                                                f"_p1_{prop1_scale}"
                                                f"_p2_{prop2_scale}"
                                                f"_p3_{prop3_scale}"
                                                f"_p4_{prop4_scale}"
                                                f"_end_{guide_scale}"
                                                f"_baseline.csv")
            guided_path = (f"_dump/results/eff_{dataset}_{category}"
                                                f"_p1_{prop1_scale}"
                                                f"_p2_{prop2_scale}"
                                                f"_p3_{prop3_scale}"
                                                f"_p4_{prop4_scale}"
                                                f"_end_{guide_scale}"
                                                f"_guided.csv")
        else:
            base_path = (f"_dump/results/{dataset}_{category}"
                                                f"_p1_{prop1_scale}"
                                                f"_p2_{prop2_scale}"
                                                f"_p3_{prop3_scale}"
                                                f"_p4_{prop4_scale}"
                                                f"_end_{guide_scale}"
                                                f"_baseline.csv")
            guided_path = (f"_dump/results/{dataset}_{category}"
                                                f"_p1_{prop1_scale}"
                                                f"_p2_{prop2_scale}"
                                                f"_p3_{prop3_scale}"
                                                f"_p4_{prop4_scale}"
                                                f"_end_{guide_scale}"
                                                f"_guided.csv")
        
    baseline_df = pd.read_csv(base_path, index_col=0, names=columns)
    guided_df = pd.read_csv(guided_path, index_col=0, names=columns)

    bmu = baseline_df.mean()
    gmu = guided_df.mean()

    bstd = baseline_df.std()
    gstd = guided_df.std()
    print("& "+category)
    print(f" & {bmu.m1:.2f} $\pm$ {bstd.m1:.2f} "
          f" & {gmu.m1:.2f} $\pm$ {gstd.m1:.2f} "
          f" & {bmu.m2:.2f} $\pm$ {bstd.m2:.2f} "
          f" & {gmu.m2:.2f} $\pm$ {gstd.m2:.2f} "
          f" & {bmu.m3:.3f} $\pm$ {bstd.m3:.3f} "
          f" & {gmu.m3:.3f} $\pm$ {gstd.m3:.3f} "
          f" & {bmu.m4:.2f} $\pm$ {bstd.m4:.2f} "
          f" & {gmu.m4:.2f} $\pm$ {gstd.m4:.3f} \\\\")
    return bmu, gmu
    # print("baseline")
    # print(baseline_median)
    # print("guided")
    # print(guided_median)

    # return baseline_median, guided_median

def box_plots(dataset, category):
    score_guide = pd.read_csv(f"_dump/results/{dataset}_{category}_p1_1.0_p2_1.0_p3_1.0_p4_1.0_end_0.1_guided.csv", index_col=0, header=None).iloc[:,0].values.reshape(-1)
    score_train = pd.read_csv(f"_dump/results/{dataset}_{category}_train_scores.csv", index_col=0, header=None).values.reshape(-1)
    # Prepare the data for box plot
    data = [score_guide, score_train]
    labels = ['repair', 'train']
    
    # Create the box plot
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=labels, showfliers=False)
    
    # Set the font size for ticks
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    
    # Set the labels and title
    # plt.xlabel('Score Type', fontsize=16)
    plt.ylabel('Anomaly Scores', fontsize=30)
    # plt.title(f'Comparison of Score Distributions for {dataset} - {category}', fontsize=18)
    
    # Save the plot
    plt.savefig(f'_dump/results/plots/{dataset}_{category}_boxplot.png')
    plt.close()



# data = []
# # Compute medians for each category and add to the data list
# for cat in VISA_CATEGORIES:
#     try:
#         compute_stats("visa", cat)
#     except:
#         continue
#     # Append data for each metric
#     for metric in baseline_median.index:
#         data.append({
#             'Category': cat, 
#             'Metric': metric, 
#             'Baseline Median': baseline_median[metric], 
#             'Guided Median': guided_median[metric]
#         })

# # Convert the data list to a DataFrame
# df = pd.DataFrame(data)

# # Save the DataFrame to a CSV file
# output_path = '_dump/results/visa_medians.csv'
# df.to_csv(output_path, index=False)
def print_median_deltas(deltas):
    print(r"\midrule")
    print(r"& $\Delta(\uparrow)$ "
          r"& \multicolumn{2}{|c|}{\textbf{+" + f"{deltas[0]*100:.2f}" + r"\%}}"
          r"& \multicolumn{2}{c|}{\textbf{+" + f"{deltas[1]*100:.2f}" + r"\%}}"
          r"& \multicolumn{2}{c|}{\textbf{+" + f"{deltas[2]*100:.2f}" + r"\%}}"
          r"& \multicolumn{2}{c}{\textbf{+" + f"{deltas[3]*100:.2f}" + r"\%}}  \\"
          )

def compute_delta_visa(model):
    bms, gms = [], []
    for cat in VISA_CATEGORIES:
        try: 
            bmu, gmu = compute_stats(model, "visa", cat, guide_scale=0.01)
            bms.append(bmu)
            gms.append(gmu)
        except:
            continue
    deltas = {}
    for i in range(4):
        m = []
        for j in range(len(bms)):
            baseline = bms[j].iloc[i] 
            guided = gms[j].iloc[i]
            delta = (baseline - guided) / np.abs(baseline)
            m.append(delta)
        deltas[i] = np.median(m)
    print_median_deltas(deltas)
    return deltas

def compute_delta_mvtec(model):
    bms, gms = [], []
    for cat in MVTEC_CATEGORIES:
        try:
            bmu, gmu = compute_stats(model, "mvtec", cat, guide_scale=0.01)
            bms.append(bmu)
            gms.append(gmu)
        except:
            continue
    deltas = {}
    for i in range(4):
        m = []
        for j in range(len(bms)):
            baseline = bms[j].iloc[i] 
            guided = gms[j].iloc[i]
            delta = (baseline - guided) / np.abs(baseline)
            m.append(delta)
        deltas[i] = np.median(m)
    print_median_deltas(deltas)
    return deltas