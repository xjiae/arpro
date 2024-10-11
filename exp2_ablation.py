import csv
import time
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from torch.utils.data import DataLoader, SubsetRandomSampler

from ad.models import *
from fixer import *
from mydatasets import *

plt.style.use("seaborn-v0_8")
PROP_SCALE_RANGE = [0.0, 0.25, 0.5, 0.75, 1.0]
END_SCALE_RANGE = list(range(1,11))
model_kwargs = {
    "coef": 1e-2,
    "learning_rate": 5e-2
}
def save_csv(i, metrics, fp):
    file = open(fp, 'a')
    writer = csv.writer(file)
    metrics.insert(0, i)
    writer.writerow(metrics)  # Write headers
    file.close()

## test time added
def get_time(
        dataset, 
        image_size=512, 
        batch_size=1,
        noise_level=500):
    if dataset == "visa":
        category = "pcb1"
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

    # dataloader = sample(original_dataloader, 10)
    # repair config
    repair_config = VisionRepairConfig(category=category, lr=1e-5, batch_size=1)

    baseline_times = []
    guided_times = []
    cnt = 0
    for i, batch in tqdm(enumerate(dataloader)):
        bad_idxs = (batch['label'] != 0)
        if bad_idxs.sum() < 1:
            continue
        cnt += 1
        if cnt > 101:
            break
        x_bad, y, m = batch['image'][bad_idxs], batch['label'][bad_idxs], batch['mask'][bad_idxs]
        x_bad = x_bad.cuda()
        x_bad_ad_out = ad(2 * x_bad - 1)
        anom_parts = (x_bad_ad_out.alpha > x_bad_ad_out.alpha.view(x_bad.size(0),-1).quantile(0.9,dim=1).view(-1,1,1,1)).long()
        good_parts = 1 - anom_parts

        average_colors = (x_bad * anom_parts).sum(dim=(-1,-2)) / (anom_parts.sum(dim=(-1,-2)))
        x_bad_masked = (1-anom_parts) * x_bad + anom_parts * (average_colors.view(-1,3,1,1))

        start_time = time.time()
        x_fix_baseline = mydiff(x_bad_masked.cuda(), noise_level, num_inference_steps=1000, progress_bar=True)
        end_time = time.time()
        baseline_time = end_time - start_time
        save_csv(i, [baseline_time], f"_dump/results/{dataset}_time_baseline.csv")
        start_time = time.time()
        out = vision_repair(x_bad, anom_parts, ad, mydiff, repair_config, noise_level)
        end_time = time.time()
        guided_time = end_time - start_time
        save_csv(i, [guided_time], f"_dump/results/{dataset}_time_guided.csv")

        baseline_times.append(baseline_time)
        guided_times.append(guided_time)
    
    return np.median(baseline_times), np.median(guided_times)

## test time added
def get_time_time(
        dataset, 
        batch_size=16,
        noise_level=100):
   # load fixer
    path = "/home/antonxue/foo/arpro/_dump/fixer_ts_diffusion_swat_best.pt"
    mytsdiff = MyTimeDiffusionModel(feature_dim=51, window_size=100)
    model_dict = torch.load(path)['model_state_dict']
    mytsdiff.load_state_dict(model_dict)
    mytsdiff.eval().cuda();

    # load ad
    ad = GPT2ADModel()
    path = '/home/antonxue/foo/arpro/_dump/ad_gpt2_swat_best.pt'
    model_dict = torch.load(path)['model_state_dict']
    ad.load_state_dict(model_dict)
    ad.eval().cuda();

    # load dataloader
    time_ds = get_timeseries_bundle(ds_name="swat", label_choice = 'all', shuffle=False, test_batch_size=batch_size, train_has_only_goods=True)
    dataloader = time_ds['test_dataloader'] 
    time_config = TimeRepairConfig(lr=1e-5, batch_size=16)
    # threshold = get_ts_threshold(ad, train_dataloader, test_dataloader)
    threshold = torch.load("_dump/swat/threshold.pt")
    # dataloader = sample(test_dataloader, 100)
    baseline_times = []
    guided_times = []
    cnt = 0
    for i, batch in tqdm(enumerate(dataloader)):
        if cnt > 100:
            break
        x, y, m  = batch
        if y.sum() == 0:
            continue
        cnt += 1

        x_bad = x.cuda()
        x_bad_ad_out = ad(x_bad)
        anom_parts = (x_bad_ad_out.alpha > threshold.cuda()).long()
        good_parts = (1 - anom_parts).long()
        start_time = time.time()
        x_fix_baseline = mytsdiff.repair(x_bad, x_bad * good_parts, good_parts, model_kwargs=model_kwargs, noise_level=noise_level, sampling_timesteps=noise_level)
        end_time = time.time()
        baseline_time = end_time - start_time
        save_csv(i, [baseline_time], f"_dump/results/{dataset}_time_baseline.csv")
        start_time = time.time()
        guided_ret = time_repair(x_bad, anom_parts, ad, mytsdiff, time_config, noise_level)
        end_time = time.time()
        guided_time = end_time - start_time
        save_csv(i, [guided_time], f"_dump/results/{dataset}_time_guided.csv")

        baseline_times.append(baseline_time)
        guided_times.append(guided_time)
    
    return np.median(baseline_times), np.median(guided_times)


### test different parameters lambda


def sample(dataloader, num_samples=50):
    dataset = dataloader.dataset
    indices = np.random.permutation(len(dataset))[:num_samples]
    sampler = SubsetRandomSampler(indices)

    # Recreate the DataLoader with the new sampler
    new_dataloader = DataLoader(dataset, batch_size=1, sampler=sampler)

    return new_dataloader

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

def detach(ad_out):
    ad_out.score = ad_out.score.detach().cpu()
    ad_out.alpha = ad_out.alpha.detach().cpu()
    ad_out.others = detach_and_move(ad_out.others)
    return ad_out

def detach_and_move(dict_tensors):
    
    for key, value in dict_tensors.items():
        
        dict_tensors[key] = [tensor.detach().cpu() for tensor in value if tensor.requires_grad]
        
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

def time_ablation(
        dataset,
        window_size=100, 
        batch_size=16,
        noise_level=50,
        steps=100
        ):
    model_kwargs = {
    "coef": 1e-2,
    "learning_rate": 5e-2   
    }
    # load fixer
    path = "/home/antonxue/foo/arpro/_dump/fixer_ts_diffusion_swat_best.pt"
    mytsdiff = MyTimeDiffusionModel(feature_dim=51, window_size=window_size)
    model_dict = torch.load(path)['model_state_dict']
    mytsdiff.load_state_dict(model_dict)
    mytsdiff.eval().cuda();

    # load ad
    ad = GPT2ADModel()
    path = '/home/antonxue/foo/arpro/_dump/ad_gpt2_swat_best.pt'
    model_dict = torch.load(path)['model_state_dict']
    ad.load_state_dict(model_dict)
    ad.eval().cuda();

    # load dataloader
    time_ds = get_timeseries_bundle(ds_name="swat", label_choice = 'all', shuffle=False, test_batch_size=batch_size, train_has_only_goods=True)
    # train_dataloader = time_ds['train_dataloader'] 
    test_dataloader = time_ds['test_dataloader'] 
    # threshold = get_ts_threshold(ad, train_dataloader, test_dataloader)
    threshold = torch.load("_dump/swat/threshold.pt")

    # dataloader = sample(test_dataloader, 10)
    # breakpoint()
    sample = 0
    for i, batch in tqdm(enumerate(test_dataloader)):
        if sample >= 101:
            break
        x, y, m  = batch
        if y.sum() == 0:
            continue
        sample += 1
        
        x_bad = x.cuda()
        x_bad_ad_out = ad(x_bad)
        anom_parts = (x_bad_ad_out.alpha > threshold.cuda()).long()
        good_parts = (1 - anom_parts).long()
        for prop1_scale in PROP_SCALE_RANGE:
            # repair config
            repair_config = TimeRepairConfig(lr=1e-5, 
                                            batch_size=batch_size, 
                                            prop1_scale=prop1_scale) 
            guided_ret = time_repair(x_bad, anom_parts, ad, mytsdiff, repair_config, noise_level, num_inference_steps=steps)
            x_fix = guided_ret['x_fix']
            x_fix_ad_out = ad(x_fix)
            metric_ours = property_metrics(x_fix, x_fix_ad_out, x_bad, x_bad_ad_out, good_parts, anom_parts)

            x_fix = x_fix.detach().cpu()
            x_fix_ad_out = detach(x_fix_ad_out)
            
            save_csv(i, metric_ours.tolist(), f"_dump/results/ablation/{dataset}"
                                                            f"_p1_{prop1_scale}.csv")
        for prop2_scale in PROP_SCALE_RANGE:
            # repair config
            repair_config = TimeRepairConfig(lr=1e-5, 
                                            batch_size=batch_size, 
                                            prop2_scale=prop2_scale) 
            guided_ret = time_repair(x_bad, anom_parts, ad, mytsdiff, repair_config, noise_level, num_inference_steps=steps)
            x_fix = guided_ret['x_fix']
            x_fix_ad_out = ad(x_fix)
            metric_ours = property_metrics(x_fix, x_fix_ad_out, x_bad, x_bad_ad_out, good_parts, anom_parts)

            x_fix = x_fix.detach().cpu()
            x_fix_ad_out = detach(x_fix_ad_out)
            
            save_csv(i, metric_ours.tolist(), f"_dump/results/ablation/{dataset}"
                                                            f"_p2_{prop2_scale}.csv")
        for prop3_scale in PROP_SCALE_RANGE:
            # repair config
            repair_config = TimeRepairConfig(lr=1e-5, 
                                            batch_size=batch_size, 
                                            prop3_scale=prop3_scale) 
            guided_ret = time_repair(x_bad, anom_parts, ad, mytsdiff, repair_config, noise_level, num_inference_steps=steps)
            x_fix = guided_ret['x_fix']
            x_fix_ad_out = ad(x_fix)
            metric_ours = property_metrics(x_fix, x_fix_ad_out, x_bad, x_bad_ad_out, good_parts, anom_parts)

            x_fix = x_fix.detach().cpu()
            x_fix_ad_out = detach(x_fix_ad_out)
            
            save_csv(i, metric_ours.tolist(), f"_dump/results/ablation/{dataset}"
                                                            f"_p3_{prop3_scale}.csv")
        for prop4_scale in PROP_SCALE_RANGE:
            # repair config
            repair_config = TimeRepairConfig(lr=1e-5, 
                                            batch_size=batch_size, 
                                            prop4_scale=prop4_scale) 
            guided_ret = time_repair(x_bad, anom_parts, ad, mytsdiff, repair_config, noise_level, num_inference_steps=steps)
            x_fix = guided_ret['x_fix']
            x_fix_ad_out = ad(x_fix)
            metric_ours = property_metrics(x_fix, x_fix_ad_out, x_bad, x_bad_ad_out, good_parts, anom_parts)

            x_fix = x_fix.detach().cpu()
            x_fix_ad_out = detach(x_fix_ad_out)
            
            save_csv(i, metric_ours.tolist(), f"_dump/results/ablation/{dataset}"
                                                            f"_p4_{prop4_scale}.csv")
        for end in END_SCALE_RANGE:
            # repair config
            repair_config = TimeRepairConfig(lr=1e-5, 
                                            batch_size=batch_size, 
                                            guide_scale=end) 
            guided_ret = time_repair(x_bad, anom_parts, ad, mytsdiff, repair_config, noise_level, num_inference_steps=steps)
            x_fix = guided_ret['x_fix']
            x_fix_ad_out = ad(x_fix)
            metric_ours = property_metrics(x_fix, x_fix_ad_out, x_bad, x_bad_ad_out, good_parts, anom_parts)


            x_fix = x_fix.detach().cpu()
            x_fix_ad_out = detach(x_fix_ad_out)
            
            save_csv(i, metric_ours.tolist(), f"_dump/results/ablation/{dataset}"
                                                            f"_control_{end}.csv")

def vision_ablation(
        dataset, 
        category, 
        image_size=512, 
        batch_size=1,
        noise_level=500
        
):
    # load ad model
    ad = FastflowAdModel(image_size=image_size)
    state_dict = torch.load(f"_dump/ad_noisy_fast_wide_resnet50_2_{dataset}_{category}_{image_size}_best.pt")["model_state_dict"]
    ad.load_state_dict(state_dict)
    ad.eval().cuda();
    # load dataloader
    original_dataloader = get_fixer_dataloader(dataset, batch_size=batch_size, category=category, split="test", image_size=image_size)

    # load fixer model
    model_path = f"_dump/fixer_diffusion_{dataset}_{category}_best.pt"
    model_dict = torch.load(model_path)['model_state_dict']
    mydiff = MyDiffusionModel(image_size=image_size)
    mydiff.load_state_dict(model_dict)
    mydiff.eval().cuda();

    dataloader = sample(original_dataloader, 10)
    
    for i, batch in tqdm(enumerate(dataloader)):
        bad_idxs = (batch['label'] != 0)
        if bad_idxs.sum() < 1:
            continue
        x_bad, y, m = batch['image'][bad_idxs], batch['label'][bad_idxs], batch['mask'][bad_idxs]
        x_bad = x_bad.cuda()
        x_bad_ad_out = ad(2 * x_bad - 1)
        anom_parts = (x_bad_ad_out.alpha > x_bad_ad_out.alpha.view(x_bad.size(0),-1).quantile(0.9,dim=1).view(-1,1,1,1)).long()
        good_parts = 1 - anom_parts
        for prop4_scale in PROP_SCALE_RANGE:
            # repair config
            repair_config = VisionRepairConfig(category=category, 
                                                lr=1e-5, 
                                                batch_size=batch_size, 
                                                prop4_scale=prop4_scale) 
            out = vision_repair(x_bad, anom_parts, ad, mydiff, repair_config, noise_level)
            
            x_fix = out['x_fix']
            x_fix_ad_out = ad(2 * x_fix - 1)
            x_fix = x_fix.clamp(0,1)
            metric_ours = property_metrics(x_fix, x_fix_ad_out, x_bad, x_bad_ad_out, good_parts, anom_parts)
            

            x_fix = x_fix.detach().cpu()
            x_fix_ad_out = detach(x_fix_ad_out)
            
            save_csv(i, metric_ours.tolist(), f"_dump/results/ablation/{dataset}_{category}"
                                                            f"_p4_{prop4_scale}.csv")
       
        for prop1_scale in PROP_SCALE_RANGE:
            # repair config
            repair_config = VisionRepairConfig(category=category, 
                                                lr=1e-5, 
                                                batch_size=batch_size, 
                                                prop1_scale=prop1_scale) 
            out = vision_repair(x_bad, anom_parts, ad, mydiff, repair_config, noise_level)
            x_fix = out['x_fix']
            x_fix_ad_out = ad(2 * x_fix - 1)
            x_fix = x_fix.clamp(0,1)
            metric_ours = property_metrics(x_fix, x_fix_ad_out, x_bad, x_bad_ad_out, good_parts, anom_parts)

            
            x_fix = x_fix.detach().cpu()
            x_fix_ad_out = detach(x_fix_ad_out)
            
            save_csv(i, metric_ours.tolist(), f"_dump/results/ablation/{dataset}_{category}"
                                                            f"_p1_{prop1_scale}.csv")
        for prop2_scale in PROP_SCALE_RANGE:
            # repair config
            repair_config = VisionRepairConfig(category=category, 
                                                lr=1e-5, 
                                                batch_size=batch_size, 
                                                prop2_scale=prop2_scale) 
            out = vision_repair(x_bad, anom_parts, ad, mydiff, repair_config, noise_level)
            x_fix = out['x_fix']
            x_fix_ad_out = ad(2 * x_fix - 1)
            x_fix = x_fix.clamp(0,1)
            metric_ours = property_metrics(x_fix, x_fix_ad_out, x_bad, x_bad_ad_out, good_parts, anom_parts)

            x_fix = x_fix.detach().cpu()
            x_fix_ad_out = detach(x_fix_ad_out)
            
            save_csv(i, metric_ours.tolist(), f"_dump/results/ablation/{dataset}_{category}"
                                                            f"_p2_{prop2_scale}.csv")
        for prop3_scale in PROP_SCALE_RANGE:
            # repair config
            repair_config = VisionRepairConfig(category=category, 
                                                lr=1e-5, 
                                                batch_size=batch_size, 
                                                prop3_scale=prop3_scale) 
            out = vision_repair(x_bad, anom_parts, ad, mydiff, repair_config, noise_level)
            x_fix = out['x_fix']
            x_fix_ad_out = ad(2 * x_fix - 1)
            x_fix = x_fix.clamp(0,1)
            metric_ours = property_metrics(x_fix, x_fix_ad_out, x_bad, x_bad_ad_out, good_parts, anom_parts)

            x_fix = x_fix.detach().cpu()
            x_fix_ad_out = detach(x_fix_ad_out)
            
            save_csv(i, metric_ours.tolist(), f"_dump/results/ablation/{dataset}_{category}"
                                                            f"_p3_{prop3_scale}.csv")
        for end in END_SCALE_RANGE:
            # repair config
            repair_config = VisionRepairConfig(category=category, 
                                                lr=1e-5, 
                                                batch_size=batch_size, 
                                                guide_scale=end) 
            out = vision_repair(x_bad, anom_parts, ad, mydiff, repair_config, noise_level)
            x_fix = out['x_fix']
            x_fix_ad_out = ad(2 * x_fix - 1)
            x_fix = x_fix.clamp(0,1)
            metric_ours = property_metrics(x_fix, x_fix_ad_out, x_bad, x_bad_ad_out, good_parts, anom_parts)

            x_fix = x_fix.detach().cpu()
            x_fix_ad_out = detach(x_fix_ad_out)
            
            save_csv(i, metric_ours.tolist(), f"_dump/results/ablation/{dataset}_{category}"
                                                            f"_end_{end}.csv")
        

def plot_vision_ablation(category):
    dataset = "visa"
    columns = ['m1', 'm2', 'm3', 'm4']
    labels = [r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$', r'$\lambda_4$', r'$\lambda_\phi$']
    metrics = ['m1', 'm2', 'm3', 'm4']

    def load_data(prop, phase):
        if phase == -1:
            df = pd.read_csv(f"_dump/results/ablation/{dataset}_{category}_end_{prop}.csv", index_col=0, names=columns)
        else:
            if prop == 1.0:
                # add others to here as well and take the median
                dfs = []
                for p in range(1,4):
                    df = pd.read_csv(f"_dump/results/ablation/{dataset}_{category}_p{p}_1.0.csv", index_col=0, names=columns)
                    dfs.append(df)
                df = pd.concat(dfs)
            else:
                df = pd.read_csv(f"_dump/results/ablation/{dataset}_{category}_p{phase}_{prop}.csv", index_col=0, names=columns)
            
        median = df.median().values
        return median
    
    def collect_data():
        p1, p2, p3, p4, end = {}, {}, {}, {}, {}
        for prop in PROP_SCALE_RANGE:
            p1[prop] = load_data(prop, 1)
            p2[prop] = load_data(prop, 2)
            p3[prop] = load_data(prop, 3)
            p4[prop] = load_data(prop, 4)
        for prop in END_SCALE_RANGE:    
            end[prop] = load_data(prop, -1)
        return p1, p2, p3, p4, end

    p1, p2, p3, p4, end = collect_data()
    
    def plot_and_save(data_dicts, metric_idx, metric_name):
        plt.figure()
        ax = plt.gca()
        props = list(data_dicts[0].keys())
        for idx, data in enumerate(data_dicts):
            medians = [data[prop][metric_idx] for prop in props]
            if 'phi' in metric_name:
                plt.plot(props, medians, label=labels[-1], marker='o')
            else:
                plt.plot(props, medians, label=labels[idx], marker='o')
        # Define custom y-ticks
        y_min, y_max = plt.ylim()
        # x_min, x_max = plt.xlim()
        y_ticks = np.linspace(y_min, y_max, num=3)
        if 'phi' in metric_name:
            x_ticks = END_SCALE_RANGE
        else:
            x_ticks = PROP_SCALE_RANGE
        # x_ticks = np.linspace(x_min, x_max, num=5)
        # plt.xlabel('Property Scale', fontsize=20)
        # plt.ylabel('Value', fontsize=20)
        plt.xticks(ticks=x_ticks, fontsize=30)
        plt.yticks(ticks=y_ticks, fontsize=30)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        plt.legend(loc='center left', bbox_to_anchor=(0.95, 0.5), fontsize=25)
        plt.tight_layout()
        plt.savefig(f'_dump/results/ablation/{dataset}_{category}_{metric_name}.png')
        plt.close()
    
    data_dicts = [p1, p2, p3, p4]
    for idx, metric in enumerate(metrics):
        plot_and_save(data_dicts, idx, metric)
    for idx, metric in enumerate(metrics):
        metric += "_phi"
        plot_and_save([end], idx, metric)
    

def plot_time_ablation():
    dataset = "swat"
    columns = ['m1', 'm2', 'm3', 'm4']
    labels = [r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$', r'$\lambda_4$', r'$\lambda_\phi$']
    metrics = ['m1', 'm2', 'm3', 'm4']

    def load_data(prop, phase):
        if phase == -1:
            df = pd.read_csv(f"_dump/results/ablation/{dataset}_control_{prop}.csv", index_col=0, names=columns)
        else:
            if prop == 1.0:
                # add others to here as well and take the median
                dfs = []
                for p in range(1,4):
                    df = pd.read_csv(f"_dump/results/ablation/{dataset}_p{p}_1.0.csv", index_col=0, names=columns)
                    dfs.append(df)
                df = pd.concat(dfs)
            else:
                df = pd.read_csv(f"_dump/results/ablation/{dataset}_p{phase}_{prop}.csv", index_col=0, names=columns)
            
        median = df.median().values
        return median
    
    def collect_data():
        p1, p2, p3, p4, end = {}, {}, {}, {}, {}
        for prop in PROP_SCALE_RANGE:
            p1[prop] = load_data(prop, 1)
            p2[prop] = load_data(prop, 2)
            p3[prop] = load_data(prop, 3)
            p4[prop] = load_data(prop, 4)
        for prop in END_SCALE_RANGE:    
            end[prop] = load_data(prop, -1)
        return p1, p2, p3, p4, end

    p1, p2, p3, p4, end = collect_data()
    
    def plot_and_save(data_dicts, metric_idx, metric_name):
        plt.figure()
        ax = plt.gca()
        props = list(data_dicts[0].keys())
        for idx, data in enumerate(data_dicts):
            medians = [data[prop][metric_idx] for prop in props]
            if 'phi' in metric_name:
                plt.plot(props, medians, label=labels[-1], marker='o')
            else:
                plt.plot(props, medians, label=labels[idx], marker='o')
        # Define custom y-ticks
        y_min, y_max = plt.ylim()
        # x_min, x_max = plt.xlim()
        y_ticks = np.linspace(y_min, y_max, num=3)
        if 'phi' in metric_name:
            x_ticks = END_SCALE_RANGE
        else:
            x_ticks = PROP_SCALE_RANGE
        # x_ticks = np.linspace(x_min, x_max, num=5)
        # plt.xlabel('Property Scale', fontsize=20)
        # plt.ylabel('Value', fontsize=20)
        plt.xticks(ticks=x_ticks, fontsize=30)
        plt.yticks(ticks=y_ticks, fontsize=30)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.legend(loc='center left', bbox_to_anchor=(0.95, 0.5), fontsize=25)
        plt.tight_layout()
        plt.savefig(f'_dump/results/ablation/{dataset}_{metric_name}.png')
        plt.close()
    
    data_dicts = [p1, p2, p3, p4]
    for idx, metric in enumerate(metrics):
        plot_and_save(data_dicts, idx, metric)
    for idx, metric in enumerate(metrics):
        metric += "_phi"
        plot_and_save([end], idx, metric)
    


# def plot_time_ablation(dataset):
#     columns = ['m1', 'm2', 'm3', 'm4']
#     labels = [r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$', r'$\lambda_4$', r'$\lambda_\phi$']
#     metrics = ['m1', 'm2', 'm3', 'm4']

#     def load_data(prop, phase):
#         df = pd.read_csv(f"_dump/results/ablation/{dataset}_p{phase}_{prop}.csv", index_col=0, names=columns)
#         median = df.mean().values
#         ci = 1.96 * df.std().values / np.sqrt(len(df))
#         return median, ci
    
#     def collect_data():
#         p1, p2, p3, p4, end = {}, {}, {}, {}, {}
#         p1_ci, p2_ci, p3_ci, p4_ci, end_ci = {}, {}, {}, {}, {}
#         for prop in PROP_SCALE_RANGE:
#             p1[prop], p1_ci[prop] = load_data(prop, 1)
#             p2[prop], p2_ci[prop] = load_data(prop, 2)
#             p3[prop], p3_ci[prop] = load_data(prop, 3)
#             p4[prop], p4_ci[prop] = load_data(prop, 4)
#             end[prop], end_ci[prop] = load_data(prop, 2)
#         return (p1, p1_ci), (p2, p2_ci), (p3, p3_ci), (p4, p4_ci), (end, end_ci)

#     (p1, p1_ci), (p2, p2_ci), (p3, p3_ci), (p4, p4_ci), (end, end_ci) = collect_data()
    
#     def plot_and_save(data_dicts, ci_dicts, metric_idx, metric_name):
#         plt.figure()
#         props = list(data_dicts[0].keys())
#         for idx, (data, ci) in enumerate(zip(data_dicts, ci_dicts)):
#             medians = [data[prop][metric_idx] for prop in props]
#             cis = [ci[prop][metric_idx] for prop in props]
#             plt.plot(props, medians, label=labels[idx])
#             plt.fill_between(props, np.array(medians) - np.array(cis), np.array(medians) + np.array(cis), alpha=0.2)
        
#         plt.xticks(fontsize=30)
#         plt.yticks(fontsize=30)
#         plt.legend(loc='center left', bbox_to_anchor=(0.9, 0.5), fontsize=25)
#         plt.tight_layout()
#         plt.savefig(f'_dump/results/ablation/{dataset}_{metric_name}.png', bbox_inches='tight')
#         plt.close()
    
#     data_dicts = [p1, p2, p3, p4, end]
#     ci_dicts = [p1_ci, p2_ci, p3_ci, p4_ci, end_ci]
#     for idx, metric in enumerate(metrics):
#         plot_and_save(data_dicts, ci_dicts, idx, metric)    
    
            
# get_time_time("swat")