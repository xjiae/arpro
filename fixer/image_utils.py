import torch
import torch.nn as nn
from collections import defaultdict
import matplotlib.pyplot as plt

def make_blobs(N, H, W, iters=512, kernel_size=3, alpha=0.5, q=0.9, device="cpu"):
    z = torch.randn(N, 1, H, W).to(device)
    a2d = nn.AvgPool2d(kernel_size, 1, kernel_size // 2).to(device)
    for _ in range(iters):
        z = alpha*z + (1-alpha) * a2d(z)
    mask = (z > z.view(N,-1).quantile(q, dim=1).view(N,1,1,1)).long() # (N,1,H,W)
    return mask


def calculate_averages(dictionary):
    return {k: sum(v) / len(v) for k, v in dictionary.items()}

def prop_loss_plot(noise_level=900, mvtec_category="transistor", num_sample=5):
    
    model_path = "../_dump/fixer_diffusion_mvtec_transistor_best.pt"
    model_dict = torch.load(model_path)['model_state_dict']
    mydiff = MyDiffusionModel()
    mydiff.load_state_dict(model_dict)
    mydiff.eval().cuda()

    ad = FastflowAdModel()
    state_dict = torch.load(f"../_dump/ad_fast_mvtec_transistor_best.pt")["model_state_dict"]
    ad.load_state_dict(state_dict)
    ad.eval().cuda()

    torch.manual_seed(1234)
    dataloader = get_fixer_dataloader("mvtec", batch_size=8, category="transistor", split="test")
    for batch in dataloader:
        break
    x_bad = batch["image"][[3,7]].cuda()
   
    # end_scales = [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
    end_scales = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 3e-1, 5e-1, 1.0]
    prop_losses = defaultdict(list)
    l1_losses = defaultdict(list)
    l2_losses = defaultdict(list)
    l3_losses = defaultdict(list)
    l4_losses = defaultdict(list)
    for s in range(num_sample):
        for end in end_scales:
            config = VisionRepairConfig(mvtec_category=mvtec_category, lr=1e-5, batch_size=2, guide_scale_end=end)
            out = vision_repair(x_bad, ad, mydiff, config, noise_level)
            prop_losses[end].append(out['prop_loss'].item())
            l1_losses[end].append(out['l1'].item())
            l2_losses[end].append(out['l2'].item())
            l3_losses[end].append(out['l3'].item())
            l4_losses[end].append(out['l4'].item())

            x_fix = out['x_fix']
            x_fix = (x_fix+1) * 0.5
            x_fix = x_fix.clamp(0,1).detach().cpu()
            plt.clf()
            fig, ax = plt.subplots(2,2)
            ax[0,0].imshow(x_bad[0].cpu().numpy().transpose(1,2,0))
            ax[0,1].imshow(x_bad[1].cpu().numpy().transpose(1,2,0))
            ax[1,0].imshow(x_fix[0].numpy().transpose(1,2,0))
            ax[1,1].imshow(x_fix[1].numpy().transpose(1,2,0))
            end_num = math.log10(end) if end > 0 else 0
            plt.savefig(config.image_folder+f"/iter{s}_end{end_num:.2f}.png")
            plt.close()

            # Save dictionaries as JSON after each update
            json_path = config.image_folder + f"/loss_data_iter{s}.json"
            with open(json_path, 'w') as f:
                json.dump({
                    'prop_losses': prop_losses,
                    'l1_losses': l1_losses,
                    'l2_losses': l2_losses,
                    'l3_losses': l3_losses,
                    'l4_losses': l4_losses
                }, f)
    plot(json_path)


def plot(file_path):
    with open(file_path, 'r') as file:
        all_losses = json.load(file)
        prop_losses = all_losses['prop_losses']
        l1_losses = all_losses['l1_losses']
        l2_losses = all_losses['l2_losses']
        l3_losses = all_losses['l3_losses']
        l4_losses = all_losses['l4_losses']
    avg_of_prop_losses = calculate_averages(prop_losses)
    avg_of_l1_losses = calculate_averages(l1_losses)
    avg_of_l2_losses = calculate_averages(l2_losses)
    avg_of_l3_losses = calculate_averages(l3_losses)
    avg_of_l4_losses = calculate_averages(l4_losses)
    
    
    fig, ax = plt.subplots()
    ax.plot(avg_of_prop_losses.keys(), avg_of_prop_losses.values(), label='Prop Losses', marker='o')
    ax.set_xlabel('End Scale')
    ax.set_ylabel('Average Loss')
    ax.legend()
    plt.title('Comparison of Average Losses')
    plt.savefig(f'/home/antonxue/foo/arpro/_dump/edit/average_prop_losses_plot.png') 
    plt.close()

    fig, ax = plt.subplots(2, 2,  figsize=(10, 6))

    ax[0, 0].plot(avg_of_l1_losses.keys(), avg_of_l1_losses.values(), label='l1 Losses', marker='o')
    ax[0, 0].set_ylabel('Average Loss')
    ax[0, 0].legend()

    ax[0, 1].plot(avg_of_l2_losses.keys(), avg_of_l2_losses.values(), label='l2 Losses', marker='o')
    ax[0, 1].set_ylabel('Average Loss')
    ax[0, 1].legend()

    ax[1, 0].plot(avg_of_l3_losses.keys(), avg_of_l3_losses.values(), label='l3 Losses', marker='o')
    ax[1, 0].set_ylabel('Average Loss')
    ax[1, 0].legend()

    ax[1, 1].plot(avg_of_l4_losses.keys(), avg_of_l4_losses.values(), label='l4 Losses', marker='o')
    ax[1, 1].set_ylabel('Average Loss')
    ax[1, 1].legend()


    # plt.title('Comparison of Losses')
    plt.savefig(f'/home/antonxue/foo/arpro/_dump/edit/average_all_losses_plot.png')  # Save the plot
        
# plot("/home/antonxue/foo/arpro/_dump/edit/loss_data_iter4.json")
# prop_loss_plot(num_sample=10)