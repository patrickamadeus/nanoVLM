import argparse
import os
import json
import torch
from models.vision_language_model import VisionLanguageModel
from models.dual_tower.dual_tower import DualTowerVLM
from train_utils.config_loader import load_yaml_config

def main():
    parser = argparse.ArgumentParser(description="Run lmms-eval on a model checkpoint.")
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file.')
    parser.add_argument('--mode', type=str, default=None, choices=['nanovlm', 'dualtower'], help='Evaluation model mode.')
    parser.add_argument('--checkpoint_path', type=str, default=None, help="Path to the model checkpoint directory.")
    parser.add_argument('--global_step', type=int, default=None, help="Global step at which the checkpoint was saved.")
    parser.add_argument('--run_name', type=str, default=None, help="The name of the training run.")

    # These arguments are based on TrainConfig, passed from the eval.slurm script
    parser.add_argument('--tasks', type=str, default=None, help='Tasks for lmms-eval, comma-separated.')
    parser.add_argument('--limit', type=int, default=None, help='Limit for lmms-eval.')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for lmms-eval.')
    
    args = parser.parse_args()
    yaml_cfg = {}
    if args.config is not None:
        loaded = load_yaml_config(args.config)
        yaml_cfg = loaded.get("evaluation", loaded)

    mode = args.mode if args.mode is not None else yaml_cfg.get("mode", "nanovlm")
    checkpoint_path = (
        args.checkpoint_path
        if args.checkpoint_path is not None
        else yaml_cfg.get("checkpoint_path", yaml_cfg.get("model"))
    )
    global_step = args.global_step if args.global_step is not None else int(yaml_cfg.get("global_step", 0))
    run_name = args.run_name if args.run_name is not None else yaml_cfg.get("run_name", "eval_run")
    tasks = args.tasks if args.tasks is not None else yaml_cfg.get("tasks", "mmstar,mmmu,ocrbench,textvqa")
    limit = args.limit if args.limit is not None else yaml_cfg.get("limit")
    batch_size = args.batch_size if args.batch_size is not None else int(yaml_cfg.get("batch_size", 128))

    if checkpoint_path is None:
        raise ValueError("checkpoint_path must be provided via CLI or YAML config.")

    from evaluation import cli_evaluate
    if mode == "dualtower":
        model = DualTowerVLM.from_pretrained(checkpoint_path)
    else:
        model = VisionLanguageModel.from_pretrained(checkpoint_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print("Running lmms-eval...")
    eval_args = argparse.Namespace(
        mode=mode,
        model=model,
        tasks=tasks,
        limit=limit,
        batch_size=batch_size,
        process_with_media=True,
        device=device,
    )
    
    eval_results = cli_evaluate(eval_args)

    output_data = {
        'global_step': global_step,
        'results': {}
    }

    if eval_results is not None and "results" in eval_results[0]:
        print("Processing evaluation results.")
        for task_name, task_results in eval_results[0]["results"].items():
            for metric_name, metric_value in task_results.items():
                if isinstance(metric_value, (int, float)):
                    key = f"{task_name}_{metric_name.split(',')[0]}"
                    output_data['results'][key] = metric_value
    else:
        print("No evaluation results to process.")

    output_dir = os.path.join('eval_results', run_name)
    os.makedirs(output_dir, exist_ok=True)
    sanitized_tasks = tasks.replace("/", "_")
    output_path = os.path.join(output_dir, f'step_{global_step}_{sanitized_tasks}.json')
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    print(f"Evaluation results for step {global_step} saved to {output_path}")

if __name__ == "__main__":
    main() 
