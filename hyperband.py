import subprocess
import numpy as np
import os
import json
import shutil
import logging
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# TODO: change L:1591 or instead 1726 or in scheduler.h train_num_batches to 19000 so that model decays properly
# TODO: turn off hellaswag eval for hyperband, and turn off generation of text after training L; 1811
# TODO: fix step printing in logging

@dataclass
class TrainingResult:
    """Class to store training results"""
    val_loss: float
    hellaswag_acc: float
    checkpoint_path: str
    training_time: float  # in seconds
    max_lr: float
    total_iters: int

@dataclass
class TrainingConfig:
    # Required flags (Table 1)
    learning_rate: float = 6e-4         # -l
    output_dir: str = "out"             # -o
    max_steps: int = -1                 # -x
    
    # Model architecture flags (Table 2)
    # model_depth: int = 6                # -ld
    # model_channels: int = 384           # -lc
    # model_heads: int = 6                # -lh
    checkpoint_every: int = 0           # -n
    resume: int = 0                     # -y
    checkpoint_path: Optional[str] = None  # -e
    
    # Data paths (Table 3)
    # train_data: str = "data/fineweb_train_*.bin"  # -id
    # val_data: str = "data/fineweb_val_*.bin"      # -iv
    # tokenizer_path: str = "data/gpt2_tokenizer.bin"  # -it
    # hellaswag_path: str = "data/hellaswag_val.bin"  # -ih
    # batch_size: int = 64                # -b
    
    # Training parameters (Table 4)
    # lr_scheduler: str = "cosine"        # -k
    # warmup_iters: int = 700            # -u
    # final_lr_frac: float = 0.0         # -q
    # weight_decay: float = 0.1          # -c
    # total_batch_size: int = 524288     # -d
    
    # Evaluation parameters (Table 5)
    # val_every: int = 250               # -v
    # sample_every: int = 20000          # -s
    run_hellaswag: int = 0             # -h

    def to_dict(self) -> dict:
        """Convert config to a readable dictionary format"""
        return {k: str(v) if isinstance(v, (str, int, float)) else None 
                for k, v in self.__dict__.items()}

    def to_cmd_args(self) -> List[str]:
        """Convert config to llm.c command line arguments"""
        args = []
        
        # Required flags
        args.extend(['-l', str(self.learning_rate)])
        args.extend(['-o', self.output_dir])
        args.extend(['-x', str(self.max_steps)])
        
        # Model architecture
        # args.extend(['-ld', str(self.model_depth)])
        # args.extend(['-lc', str(self.model_channels)])
        # args.extend(['-lh', str(self.model_heads)])
        args.extend(['-n', str(self.checkpoint_every)])
        args.extend(['-y', str(self.resume)])
        if self.checkpoint_path:
            args.extend(['-e', self.checkpoint_path])
        
        # Data paths
        # args.extend(['-id', self.train_data])
        # args.extend(['-iv', self.val_data])
        # args.extend(['-it', self.tokenizer_path])
        # args.extend(['-ih', self.hellaswag_path])
        # args.extend(['-b', str(self.batch_size)])
        
        # Training parameters
        # args.extend(['-k', self.lr_scheduler])
        # args.extend(['-u', str(self.warmup_iters)])
        # args.extend(['-q', str(self.final_lr_frac)])
        # args.extend(['-c', str(self.weight_decay)])
        # args.extend(['-d', str(self.total_batch_size)])
        
        # Evaluation
        # args.extend(['-v', str(self.val_every)])
        # args.extend(['-s', str(self.sample_every)])
        args.extend(['-h', str(self.run_hellaswag)])
        
        return args

def setup_logging(base_dir: str) -> logging.Logger:
    """Setup logging configuration"""
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('hyperband')
    logger.setLevel(logging.INFO)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f'hyperband_{timestamp}.log')
    )
    console_handler = logging.StreamHandler()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class HyperbandOptimizer:
    def __init__(self, 
                 max_iter: int = 32000,
                 eta: int = 3,
                 base_dir: str = "hyperband_runs"):
        self.max_iter = max_iter
        self.eta = eta
        self.s_max = int(np.log(max_iter) / np.log(eta))
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
        self.logger = setup_logging(base_dir)
        self.logger.info(f"Initializing Hyperband with max_iter={max_iter}, eta={eta}, s_max={self.s_max}")
        
        self.history = {
            'configs': [],
            'best_loss': float('inf'),
            'best_config': None,
            'total_iterations': 0,
            'total_training_time': 0
        }
        self.round_checkpoints = {}

    def cleanup_previous_round_checkpoints(self, bracket: int, current_round: int):
        """Clean up checkpoints from previous rounds in the current bracket"""
        if bracket in self.round_checkpoints and current_round > 1:
            previous_round = current_round - 1
            if previous_round in self.round_checkpoints[bracket]:
                self.logger.info(f"Cleaning up checkpoints from bracket {bracket}, round {previous_round}")
                for checkpoint_path in self.round_checkpoints[bracket][previous_round]:
                    if os.path.exists(checkpoint_path):
                        try:
                            os.remove(checkpoint_path)
                            self.logger.info(f"Deleted checkpoint: {checkpoint_path}")
                        except Exception as e:
                            self.logger.warning(f"Failed to delete checkpoint {checkpoint_path}: {e}")
                # Clear the list after deletion
                del self.round_checkpoints[bracket][previous_round]

    def generate_random_config(self) -> TrainingConfig:
        """Generate random hyperparameters within reasonable ranges"""
        return TrainingConfig(
            learning_rate=float(np.random.uniform(1e-5, 1e-3)),
            # batch_size=int(np.random.choice([32, 64, 128])),
            # warmup_iters=int(np.random.randint(500, 1000)),
            # final_lr_frac=float(np.random.uniform(0.0, 0.2)),
            # weight_decay=float(np.random.uniform(0.05, 0.15)),
            # lr_scheduler=np.random.choice(["cosine", "linear"])
        )

    def parse_training_output(self, output: str) -> Tuple[float, float]:
        """Parse validation loss and hellaswag accuracy from training output"""
        val_loss = float('inf')
        hellaswag_acc = 0.0
        max_lr = 0.0
        
        lines = output.strip().split('\n')
        for line in reversed(lines):
            if "val loss" in line:
                try:
                    val_loss = float(line.split("val loss")[-1].strip())
                except ValueError:
                    self.logger.warning(f"Could not parse val loss from line: {line}")
            
            # if "HellaSwag:" in line:
            #     try:
            #         hellaswag_acc = float(line.split("=")[-1].strip())
            #     except ValueError:
            #         self.logger.warning(f"Could not parse hellaswag accuracy from line: {line}")

            if "step" in line and "lr" in line:
                try:
                    # Extract lr value using split on '|'
                    lr_part = [part.strip() for part in line.split('|') if 'lr' in part][0]
                    # Extract the scientific notation number
                    lr = float(lr_part.split('lr')[1].strip())
                    max_lr = max(max_lr, lr)
                    # self.logger.info(f"Step {current_iter}: Learning rate = {lr:.2e}")
                except (ValueError, IndexError):
                    self.logger.warning(f"Could not parse learning rate from line: {line}")

        
        return val_loss, hellaswag_acc, max_lr

    def train_model(self, 
                   config: TrainingConfig, 
                   config_id: str, 
                   num_iters: int,
                   previous_checkpoint: Optional[str] = None,
                   previous_iters: int = 0) -> TrainingResult:
        """Train model and track execution time"""
        
        self.logger.info(
            f"\nStarting training for config {config_id}:"
            f"\n  Iterations: {num_iters}"
            f"\n  Previous iterations: {previous_iters}"
            f"\n  Previous checkpoint: {previous_checkpoint if previous_checkpoint else 'None'}"
        )
        
        run_dir = os.path.join(self.base_dir, f"run_{config_id}")
        os.makedirs(run_dir, exist_ok=True)
        config.output_dir = run_dir
        config.max_steps = num_iters
        # config.run_hellaswag = 1
        config.run_hellaswag = 0
        
        if previous_checkpoint:
            config.checkpoint_path = previous_checkpoint
            config.resume = 1
        else:
            config.resume = 0
        
        checkpoint_path = os.path.join(run_dir, "checkpoint.bin")
        config.checkpoint_every = num_iters
        
        cmd = ['./train_gpt2cu']
        cmd.extend(config.to_cmd_args())
        
        try:
            self.logger.info(f"Running command: {' '.join(cmd)}")
            
            # Track training time
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=False,
                check=True
            )
            training_time = time.time() - start_time

            # Decode output with errors='replace' to handle invalid characters
            stdout = result.stdout.decode('utf-8', errors='replace')
            stderr = result.stderr.decode('utf-8', errors='replace')
            
            val_loss, hellaswag_acc, max_lr = self.parse_training_output(stdout)

            total_iters = num_iters
            
            self.logger.info(
                f"Training completed for config {config_id}:"
                f"\n  Training time: {timedelta(seconds=int(training_time))}"
                f"\n  Final validation loss: {val_loss:.6f}"
                f"\n  Hellaswag accuracy: {hellaswag_acc:.2%}"
                f"\n  Total iterations: {total_iters}"
                f"\n  Maximum learning rate: {max_lr:.2e}"
                f"\n  Checkpoint saved to: {checkpoint_path}"
            )
            
            return TrainingResult(val_loss, hellaswag_acc, checkpoint_path, training_time, max_lr, total_iters)
            
        except subprocess.CalledProcessError as e:
            self.logger.error(
            f"Error training config {config_id}:"
            f"\n  stdout: {e.stdout.decode('utf-8', errors='replace') if e.stdout else ''}"
            f"\n  stderr: {e.stderr.decode('utf-8', errors='replace') if e.stderr else ''}"
        )
            return TrainingResult(float('inf'), 0.0, "", 0.0, 0.0, previous_iters)

    def log_config(self, 
                  config_id: str, 
                  config: TrainingConfig, 
                  result: TrainingResult,
                  iterations: int, 
                  phase: str):
        """Log configuration details and performance"""
        config_dict = config.to_dict()
        log_entry = {
            'config_id': config_id,
            'config': config_dict,
            'val_loss': result.val_loss,
            'hellaswag_acc': result.hellaswag_acc,
            'iterations': iterations,
            'total_iterations': result.total_iters,
            'training_time': result.training_time,
            'max_lr': result.max_lr,
            'phase': phase,
            'timestamp': datetime.now().isoformat()
        }
        
        self.history['configs'].append(log_entry)
        self.history['total_training_time'] += result.training_time
        
        self.logger.info(
            f"\nConfiguration {config_id} ({phase}):"
            f"\n  Hyperparameters: {json.dumps(config_dict, indent=2)}"
            f"\n  Current iterations: {iterations}"
            f"\n  Total iterations: {result.total_iters}"
            f"\n  Training time: {timedelta(seconds=int(result.training_time))}"
            f"\n  Validation Loss: {result.val_loss:.6f}"
            f"\n  Hellaswag Accuracy: {result.hellaswag_acc:.2%}"
            f"\n  Maximum Learning Rate: {result.max_lr:.2e}"
        )

    def run_optimization(self) -> Tuple[TrainingConfig, TrainingResult]:
        start_time = datetime.now()
        self.logger.info(f"Starting Hyperband optimization at {start_time}")
        
        best_config = None
        best_result = TrainingResult(float('inf'), 0.0, "", 0.0,0.0, 0)
        total_configs = 0
        
        for s in reversed(range(self.s_max + 1)):
            n = int(np.ceil((self.s_max + 1) * (self.eta ** s) / (s + 1)))
            r = int(self.max_iter * (self.eta ** (-s)))

            self.round_checkpoints[s] = {}
            
            self.logger.info(
                f"\nBracket {s}:"
                f"\n  Initial configurations: {n}"
                f"\n  Initial iterations: {r}"
            )
            
            configs: Dict[str, Dict] = {}
            for i in range(n):
                config_id = f"s{s}_n{i}"
                configs[config_id] = {
                    'config': self.generate_random_config(),
                    'result': None,
                    'checkpoint': None,
                    'total_iters': 0
                }
                total_configs += 1
            
            for i in range(s + 1):
                # self.cleanup_previous_round_checkpoints(s, i)

                n_i = int(n * (self.eta ** (-i)))
                r_i = int(r * (self.eta ** i))

                self.round_checkpoints[s][i] = []
                
                self.logger.info(
                    f"\nBracket {s}, Round {i}:"
                    f"\n  Active configs: {len(configs)}"
                    f"\n  Iterations: {r_i}"
                )
                
                for config_id, config_data in configs.items():
                    result = self.train_model(
                        config_data['config'],
                        config_id,
                        r_i,
                        config_data['checkpoint'],
                        config_data.get('total_iters', 0) 
                    )
                    
                    configs[config_id]['result'] = result
                    configs[config_id]['checkpoint'] = result.checkpoint_path
                    configs[config_id]['total_iters'] = result.total_iters
                    
                    if result.checkpoint_path:
                        self.round_checkpoints[s][i].append(result.checkpoint_path)

                    self.log_config(
                        config_id, 
                        config_data['config'], 
                        result,
                        r_i, 
                        f'bracket_{s}_round_{i}'
                    )
                    
                    if result.val_loss < best_result.val_loss:
                        best_result = result
                        best_config = config_data['config']
                        self.logger.info(
                            f"\nNew best configuration found!"
                            f"\n  Config ID: {config_id}"
                            f"\n  Validation Loss: {result.val_loss:.6f}"
                            f"\n  Hellaswag Accuracy: {result.hellaswag_acc:.2%}"
                            f"\n  Training Time: {timedelta(seconds=int(result.training_time))}"
                        )
                
                if i < s:
                    sorted_configs = sorted(
                        configs.items(),
                        key=lambda x: x[1]['result'].val_loss
                    )
                    n_survive = int(n_i / self.eta)
                    
                    self.logger.info(
                        f"\nEliminating {len(configs) - n_survive} configurations:"
                        f"\n  Surviving: {n_survive}"
                    )
                    
                    surviving_configs = dict(sorted_configs[:n_survive])
                    
                    for config_id in configs.keys() - surviving_configs.keys():
                        run_dir = os.path.join(self.base_dir, f"run_{config_id}")
                        if os.path.exists(run_dir):
                            shutil.rmtree(run_dir)
                    
                    configs = surviving_configs
            # cleanup final round ckpts for this bracket
            # self.cleanup_previous_round_checkpoints(s, s + 1)
        
        end_time = datetime.now()
        duration = end_time - start_time
        total_training_time = timedelta(seconds=int(self.history['total_training_time']))

        # Final summary logging
        self.logger.info(
            f"\nHyperband optimization completed:"
            f"\n  Total Duration: {duration}"
            f"\n  Total Training Time: {total_training_time}"
            f"\n  Total Configurations Tried: {total_configs}"
            f"\n  Best Validation Loss: {best_result.val_loss:.6f}"
            f"\n  Best Hellaswag Accuracy: {best_result.hellaswag_acc:.2%}"
            f"\n  Best Configuration Total Iterations: {best_result.total_iters}"
            f"\n  Best Configuration Parameters:"
            f"\n    Learning Rate: {best_config.learning_rate}"
            f"\n    Maximum Learning Rate: {best_result.max_lr:.2e}"
            # f"\n    Batch Size: {best_config.batch_size}"
            # f"\n    Warmup Iterations: {best_config.warmup_iters}"
            # f"\n    LR Scheduler: {best_config.lr_scheduler}"
            # f"\n    Weight Decay: {best_config.weight_decay}"
            # f"\n    Final LR Fraction: {best_config.final_lr_frac}"
        )
        
        # Save complete history with all metrics
        history_path = os.path.join(self.base_dir, "optimization_history.json")
        with open(history_path, 'w') as f:
            json.dump({
                'optimization_duration': str(duration),
                'total_training_time': str(total_training_time),
                'total_configs_tried': total_configs,
                'best_config': best_config.to_dict(),
                'best_result': {
                    'val_loss': best_result.val_loss,
                    'hellaswag_acc': best_result.hellaswag_acc,
                    'training_time': best_result.training_time
                },
                'configs': self.history['configs']
            }, f, indent=2)
        
        return best_config, best_result

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = f"hyperband_runs_{timestamp}"
    
    optimizer = HyperbandOptimizer(
        max_iter=735,  # Maximum iterations for any single training run
        eta=3,          # Reduction factor between rounds
        base_dir=base_dir
    )
    
    best_config, best_result = optimizer.run_optimization()
    
    # Save best configuration with all metrics
    best_config_path = os.path.join(base_dir, 'best_config.json')
    with open(best_config_path, 'w') as f:
        json.dump({
            'config': best_config.to_dict(),
            'performance': {
                'val_loss': best_result.val_loss,
                'hellaswag_accuracy': best_result.hellaswag_acc,
                'training_time': best_result.training_time,
                'checkpoint_path': best_result.checkpoint_path
            }
        }, f, indent=2)
    
    print(f"\nOptimization completed. Results saved in {base_dir}")
    print(f"Best configuration saved to {best_config_path}")

if __name__ == "__main__":
    main()
