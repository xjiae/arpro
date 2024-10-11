import csv
import torch
import numpy as np
from tqdm import tqdm
from ad import *
from mydatasets import *


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

def get_train_score(
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
    pbar = tqdm(enumerate(dataloader))
    for i, batch in pbar:
        x = batch["image"].cuda()
        ad_out = ad(x)
        anom = ad_out.score.mean().detach().cpu()
        save_csv(i, [anom.item()], f"_dump/results/train/{dataset}_{category}_scores.csv")
        pbar.update(1)


def get_swat_train_score(
        dataset, 
        batch_size=16):
    # load ad
    ad = GPT2ADModel()
    path = '/home/antonxue/foo/arpro/_dump/ad_gpt2_swat_best.pt'
    model_dict = torch.load(path)['model_state_dict']
    ad.load_state_dict(model_dict)
    ad.eval().cuda();
    # load dataloader
    time_ds = get_timeseries_bundle(ds_name="swat", label_choice = 'all', shuffle=False, test_batch_size=batch_size, train_has_only_goods=True)
    train_dataloader = time_ds['train_dataloader'] 
    pbar = tqdm(enumerate(train_dataloader))
    for i, batch in pbar:
        x, y, m  = batch
        if y.sum() != 0:
            continue
        x = x.cuda()
        ad_out = ad(x)
        anom = torch.norm(ad_out.score.detach().cpu().mean(), p=2)
        save_csv(i, [anom.item()], f"_dump/results/{dataset}_train_scores.csv")
        pbar.update(1)
        
def vision_verify(option="guided"):
    dataset = "visa"
    scores = []
    for category in VISA_CATEGORIES:
        # print(category)
        # get_train_score("visa", category)
        try:
            train_scores = pd.read_csv(f"_dump/results/train/{dataset}_{category}_scores.csv", index_col=0).values
            test_scores = pd.read_csv(f"_dump/results/{dataset}_{category}"
                                                    f"_p1_1.0"
                                                    f"_p2_1.0"
                                                    f"_p3_{1.0}"
                                                    f"_p4_{1.0}"
                                                    f"_end_{10.0}"
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
def time_verify(step, option="guided"):
    train_scores = pd.read_csv(f"_dump/results/swat_train_scores.csv", index_col=0).values
    test_scores = pd.read_csv(f"_dump/results/swat"
                                            f"_p1_1.0"
                                            f"_p2_1.0"
                                            f"_p3_{1.0}"
                                            f"_p4_{1.0}"
                                            f"_end_{10.0}"
                                            f"_steps_{step}"
                                            f"_{option}.csv", index_col=0).iloc[:, 0]
    error = get_ood_error(test_scores,train_scores)
    print(error)


# get_swat_train_score("swat")
# print("swat")
# time_verify("swat")

