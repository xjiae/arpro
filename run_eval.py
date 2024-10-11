import argparse
from exp1_property import *
from exp2_ablation import *

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, default="median")
    parser.add_argument("--ad", type=str, default="gpt2")
    # Dataset-specific parameters
    parser.add_argument("--dataset", type=str, default="visa")
    parser.add_argument("--category", type=str, default="pcb4")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda")

    # Hyperparameters
    parser.add_argument("--p1", type=float, default=1.0)
    parser.add_argument("--p2", type=float, default=1.0)
    parser.add_argument("--p3", type=float, default=1.0)
    parser.add_argument("--p4", type=float, default=1.0)
    parser.add_argument("--end", type=float, default=10.0)
    parser.add_argument("--noise", type=int, default=500)
    parser.add_argument("--steps", type=int, default=200)

    parser.add_argument("--output_dir",
        default=str(Path(Path(__file__).parent.resolve(), "_dump")))
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    args = parse_args()

    

    if args.dataset == "visa" or args.dataset == "mvtec":
        if args.task == "median":
            compute_stats(
                        dataset=args.dataset, 
                        category=args.category, 
                        prop1_scale=args.p1,
                        prop2_scale=args.p2,
                        prop3_scale=args.p3,
                        prop4_scale=args.p4,
                        guide_scale=args.end
                        )
        elif args.task == "exp1":
            if args.ad == "fastflow":
                eval_image_property_improvement(dataset=args.dataset, 
                                        category=args.category, 
                                        image_size=args.image_size, 
                                        noise_level=args.noise,
                                        batch_size=args.batch_size, 
                                        prop1_scale=args.p1,
                                        prop2_scale=args.p2,
                                        prop3_scale=args.p3,
                                        prop4_scale=args.p4,
                                        guide_scale=args.end
                                        )
            elif args.ad == "efficientad":
                eval_image_property_improvement_eff(dataset=args.dataset, 
                                        category=args.category, 
                                        image_size=args.image_size, 
                                        noise_level=args.noise,
                                        batch_size=args.batch_size, 
                                        prop1_scale=args.p1,
                                        prop2_scale=args.p2,
                                        prop3_scale=args.p3,
                                        prop4_scale=args.p4,
                                        guide_scale=args.end
                                        )
        elif args.task == 'exp2':
            vision_ablation(
                            dataset=args.dataset, 
                            category=args.category, 
                            image_size=args.image_size, 
                            noise_level=args.noise,
                            batch_size=args.batch_size, 
                            )

    elif args.dataset == "swat" or args.dataset == "wadi" or args.dataset == "hai":
        if args.task == "median":
            compute_stats(
                        dataset=args.dataset, 
                        prop1_scale=args.p1,
                        prop2_scale=args.p2,
                        prop3_scale=args.p3,
                        prop4_scale=args.p4,
                        guide_scale=args.end
                        )
        elif args.task == "exp1":
            if args.ad == "gpt2":
                eval_time_property_improvement(dataset=args.dataset, 
                                            noise_level=args.noise,
                                            batch_size=args.batch_size, 
                                            prop1_scale=args.p1,
                                            prop2_scale=args.p2,
                                            prop3_scale=args.p3,
                                            prop4_scale=args.p4,
                                            guide_scale=args.end,
                                            num_inference_steps=args.steps)
            elif args.ad == "llama2":
                eval_time_property_improvement_llama(dataset=args.dataset, 
                                            noise_level=args.noise,
                                            batch_size=args.batch_size, 
                                            prop1_scale=args.p1,
                                            prop2_scale=args.p2,
                                            prop3_scale=args.p3,
                                            prop4_scale=args.p4,
                                            guide_scale=args.end,
                                            num_inference_steps=args.steps)
        elif args.task == 'exp2':
            time_ablation(dataset=args.dataset, 
                          noise_level=args.noise,
                          batch_size=args.batch_size,
                          steps=args.steps)
    elif args.dataset == "webtext":
        if args.task == "median":
            compute_stats(
                        dataset=args.dataset, 
                        prop1_scale=args.p1,
                        prop2_scale=args.p2,
                        prop3_scale=args.p3,
                        prop4_scale=args.p4,
                        guide_scale=args.end
                        )
        else:
            eval_text_property_improvement(dataset=args.dataset, 
                                        noise_level=args.noise,
                                        num_inference_steps=args.steps,
                                        batch_size=args.batch_size, 
                                        prop1_scale=args.p1,
                                        prop2_scale=args.p2,
                                        prop3_scale=args.p3,
                                        prop4_scale=args.p4,
                                        guide_scale=args.end)   
