import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
### report the anomaly detector performance
from ad.models import *
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

'''(image AUROC, pixel AUROC)'''

def eval(dataset, category, image_size, batch_size):
    ad = FastflowAdModel(image_size=image_size)
    state_dict = torch.load(f"_dump/ad_noisy_fast_wide_resnet50_2_{dataset}_{category}_512_best.pt")["model_state_dict"]
    ad.load_state_dict(state_dict)
    ad.eval().cuda();
    dataloader = get_fixer_dataloader(dataset, batch_size=batch_size, category=category, split="test", image_size=image_size)
    results = {
        "y": [],
        "y_pred": [],
        "m": [],
        "m_pred": []
        }
    for batch in tqdm(dataloader):
        x_bad = batch["image"].cuda()
        ad_out = ad(x_bad)
        y, m = batch['label'].cpu().flatten(), batch['mask'].cpu().flatten()
        y_pred, m_pred = ad_out.score.detach().cpu().flatten(), ad_out.alpha.detach().cpu().flatten()
        results['y'].extend(y.tolist())
        results['y_pred'].extend(y_pred.tolist())
        results['m'].extend(m.tolist())
        results['m_pred'].extend(m_pred.tolist())
    image_auroc = roc_auc_score(results["y"], results["y_pred"])
    pixel_auroc = roc_auc_score(results["m"], results["m_pred"])
    
    return image_auroc, pixel_auroc

# print(eval("visa", "pcb4", 512, 2))
res = []
for cat in VISA_CATEGORIES:
    print(cat)
    image_auroc, pixel_auroc = eval("visa", cat, 512, 2)
    res.append({
        'Category': cat,
        'Image AUROC': image_auroc,
        'Pixel AUROC': pixel_auroc
    })
    print(f"{cat}: Image AUROC = {image_auroc}, Pixel AUROC = {pixel_auroc}")
df = pd.DataFrame(res)

# Specify the filename
filename = '_dump/results/vision_auroc_results.csv'

# Save the DataFrame to CSV
df.to_csv(filename, index=False)
print(f'Data saved to {filename}')
    




'''time AUROC'''
def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred

def eval_ts():
    time_ds = get_timeseries_bundle(ds_name="swat", label_choice = 'all', shuffle=False, test_batch_size=16, train_has_only_goods=True)
    train_dataloader = time_ds['train_dataloader'] 
    test_dataloader = time_ds['test_dataloader'] 
    model = GPT2ADModel()
    path = '/home/antonxue/foo/arpro/_dump/ad_gpt2_swat_best.pt'
    model_dict = torch.load(path)['model_state_dict']
    model.load_state_dict(model_dict)
    model.eval().cuda();
    anomaly_criterion = nn.MSELoss(reduce=False)
    results = {
        "y": [],
        "y_pred": []
        }
    attens_energy = []
    ### code adapted from https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All/blob/main/Anomaly_Detection/exp/exp_anomaly_detection.py
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_m) in enumerate(train_dataloader):
            batch_x = batch_x.float().cuda()
            # reconstruction
            outputs = model(batch_x)
            # criterion
            score = torch.mean(anomaly_criterion(batch_x, outputs.others['x_recon']), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)

    attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    train_energy = np.array(attens_energy)
    attens_energy = []
    test_labels = []
    for batch in tqdm(test_dataloader):
        x, y, _  = batch
        x = x.cuda()
        outputs = model(x)
        score = torch.mean(anomaly_criterion(x, outputs.others['x_recon']), dim=-1)
        score = score.detach().cpu().numpy()
        attens_energy.append(score)
        test_labels.append(y)
        results['y'].extend(y.flatten().tolist())
        results['y_pred'].extend(score.flatten().tolist())
    attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    test_energy = np.array(attens_energy)
    combined_energy = np.concatenate([train_energy, test_energy], axis=0)
    threshold = np.percentile(combined_energy, 99)
    print("Threshold :", threshold)
    # (3) evaluation on the test set
    pred = (test_energy > threshold).astype(int)
    test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
    test_labels = np.array(test_labels)
    gt = test_labels.astype(int)
    

    print("pred:   ", pred.shape)
    print("gt:     ", gt.shape)

    # (4) detection adjustment
    gt, pred = adjustment(gt, pred)

    pred = np.array(pred)
    gt = np.array(gt)
    print("pred: ", pred.shape)
    print("gt:   ", gt.shape)
    # auroc = roc_auc_score(results["y"], results["y_pred"])
    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
    print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
        accuracy, precision,
        recall, f_score))

    # return auroc
        
# print(eval_ts())


'''text AUROC '''

def eval_text():
    model = RobertaADModel().eval()
    dataloader = get_fixer_dataloader(dataset_name="webtext", batch_size=8, split="test")
    results = {
        "y": [],
        "score": [],
        "y_pred": []}
    for batch in tqdm(dataloader):
        x, _, y = batch
        x = x.cuda()
        # real: 1 | fake: 0
        out = model(x)
        results['y'].extend(y.tolist())
        results['score'].extend(out.score.detach().cpu().tolist())
        results['y_pred'].extend(out.others['y_pred'].detach().cpu().tolist())
    
    auroc = roc_auc_score(results["y"], results["score"])
    accuracy = accuracy_score(results['y'], results['y_pred'])
    precision, recall, f_score, support = precision_recall_fscore_support(results['y'], results['y_pred'], average='binary')
    print("Accuracy : {:0.4f}, \
          Precision : {:0.4f}, \
          Recall : {:0.4f}, \
          F-score : {:0.4f},\
          AUROC : {:0.4f} ".format(
        accuracy, precision,
        recall, f_score, auroc))