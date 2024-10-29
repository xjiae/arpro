import csv
import torch
import numpy as np
from tqdm import tqdm
from ad import *
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

def get_ood_error(
        val_scores, # softmax score for test set
        cal_scores, # softmax scores for train set
        alpha=0.05,
        ):
    n = len(cal_scores)
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n)
    # if qhat < 0:
    #     val_scores = -1 * val_scores
    y_pred = (val_scores < qhat)
    return y_pred.mean()

def save_csv(i, anom, fp):
    file = open(fp, 'a')
    writer = csv.writer(file)
    anom.insert(0, i)
    writer.writerow(anom)  # Write headers
    file.close()

def get_train_score_fastflow(
        dataset, 
        category,
        image_size=512, 
        batch_size=1):
    # load ad model
    ad = FastflowAdModel(image_size=image_size)
    state_dict = torch.load(f"_dump/ad_noisy_fast_wide_resnet50_2_{dataset}_{category}_{image_size}_best.pt")["model_state_dict"]
    ad.load_state_dict(state_dict)
    ad.eval().cuda();
    dataloader = get_fixer_dataloader(dataset, 
                                      batch_size=batch_size, 
                                      category=category, 
                                      split="train", 
                                      image_size=image_size)
    pbar = enumerate(tqdm(dataloader))
    for i, batch in pbar:
        x = batch["image"].cuda()
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        ad_out = ad(x)
        anom = ad_out.score.mean().detach().cpu()
        save_csv(i, [anom.item()], f"_dump/results/fast_train/{dataset}_{category}_scores.csv")

def get_train_score_efficientad(
        dataset, 
        category,
        image_size=512, 
        batch_size=1):
    # load ad model
    ad = EfficientAdADModel(model_size="medium")
    state_dict = torch.load(f"_dump/ad_eff_{dataset}_{category}_best.pt")["model_state_dict"]
    ad.load_state_dict(state_dict)
    ad.eval().cuda();
    dataloader = get_fixer_dataloader(dataset, 
                                      batch_size=batch_size, 
                                      category=category, 
                                      split="train", 
                                      image_size=image_size)
    pbar = enumerate(tqdm(dataloader))
    for i, batch in pbar:
        x = batch["image"].cuda()
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        ad_out = ad(x)
        anom = ad_out.score.mean().detach().cpu()
        save_csv(i, [anom.item()], f"_dump/results/eff_train/{dataset}_{category}_scores.csv")


def get_train_score(
        dataset, 
        batch_size=16, 
        num_features=51):
    # load ad
    ad = GPT2ADModel(num_features=num_features)
    path = f'/home/antonxue/foo/arpro/_dump/ad_gpt2_{dataset}_best.pt'
    model_dict = torch.load(path)['model_state_dict']
    ad.load_state_dict(model_dict)
    ad.eval().cuda();
    # load dataloader
    time_ds = get_timeseries_bundle(ds_name=dataset, label_choice = 'all', shuffle=False, test_batch_size=batch_size, train_has_only_goods=True)
    train_dataloader = time_ds['train_dataloader'] 
    pbar = enumerate(tqdm(train_dataloader))
    for i, batch in pbar:
        x, y, m  = batch
        if y.sum() != 0:
            continue
        x = x.cuda()
        ad_out = ad(x)
        anom = torch.norm(ad_out.score.detach().cpu().mean(), p=2)
        save_csv(i, [anom.item()], f"_dump/results/{dataset}_train_scores.csv")
        
def vision_verify_fastflow(option="guided", dataset="visa"):
    
    scores = []
    cat = VISA_CATEGORIES if dataset == "visa" else MVTEC_CATEGORIES
    for category in cat:
        # print(category)
        # get_train_score("visa", category)
        try:
            train_scores = pd.read_csv(f"_dump/results/fast_train/{dataset}_{category}_scores.csv", index_col=0).values
            test_scores = pd.read_csv(f"_dump/results/{dataset}_{category}"
                                                    f"_p1_1.0"
                                                    f"_p2_1.0"
                                                    f"_p3_{1.0}"
                                                    f"_p4_{1.0}"
                                                    f"_end_{0.01}"
                                                    f"_{option}.csv", index_col=0).iloc[:, 0]
            error = get_ood_error(test_scores,train_scores)
            print(f"{error:.2f}")
            scores.append(error)
            print(category)

            print("-----------------------------------")
        except:
            continue
    print(np.median(scores))
    
    return scores

def vision_verify_efficientad(option="guided", dataset="visa"):
    
    scores = []
    cat = VISA_CATEGORIES if dataset == "visa" else MVTEC_CATEGORIES
    for category in cat:
        # print(category)
        # get_train_score("visa", category)
        try:
            train_scores = pd.read_csv(f"_dump/results/eff_train/{dataset}_{category}_scores.csv", index_col=0).values
            test_scores = pd.read_csv(f"_dump/results/eff_{dataset}_{category}"
                                                    f"_p1_1.0"
                                                    f"_p2_1.0"
                                                    f"_p3_{1.0}"
                                                    f"_p4_{1.0}"
                                                    f"_end_{0.01}"
                                                    f"_{option}.csv", index_col=0).iloc[:, 0]
            error = get_ood_error(test_scores,train_scores)
            print(f"{error:.2f}")
            scores.append(error)
            print(category)

            print("-----------------------------------")
        except:
            continue
    print(np.median(scores))
    
    return scores



def time_verify(step=200, option="guided",dataset="swat"):
    train_scores = pd.read_csv(f"_dump/results/{dataset}_train_scores.csv", index_col=0).values
    test_scores = pd.read_csv(f"_dump/results/{dataset}"
                                            f"_p1_1.0"
                                            f"_p2_1.0"
                                            f"_p3_{1.0}"
                                            f"_p4_{1.0}"
                                            f"_end_{0.01}"
                                            f"_steps_{step}"
                                            f"_{option}.csv", index_col=0).iloc[:, 0]
    error = get_ood_error(test_scores,train_scores)
    print(error)


# get_swat_train_score("swat")
# print("swat")
# time_verify("swat")
# for cat in VISA_CATEGORIES:
#     get_train_score_efficientad(dataset='visa', category=cat)
# vision_verify_efficientad(option="guided", dataset="visa")


# for cat in MVTEC_CATEGORIES:
#     get_train_score_efficientad(dataset='mvtec', category=cat, image_size=256)
# vision_verify_efficientad(option="guided", dataset="mvtec")

# for cat in MVTEC_CATEGORIES:
#     get_train_score_fastflow(dataset='mvtec', category=cat, image_size=256)
# vision_verify_fastflow(option="guided", dataset="mvtec")
# vision_verify_fastflow(option="baseline", dataset="mvtec")
# get_train_score(dataset="hai", num_features=86)
# get_train_score(dataset="wadi", num_features=127)
time_verify(dataset="hai", option="guided")
time_verify(dataset="hai", option="baseline")
time_verify(dataset="wadi", option="guided")
time_verify(dataset="wadi", option="baseline")