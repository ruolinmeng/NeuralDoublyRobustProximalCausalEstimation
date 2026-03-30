import argparse
import torch

def get_config():
    parser = argparse.ArgumentParser(description="Causal Inference Experiments")

    # General
    parser.add_argument("--run_name", type=str, default="bridges",
                        help="Name of the run (used in saving results/checkpoints)")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20, 50)),
                        help="List of seeds for experiments")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Torch device, e.g. 'cpu', 'cuda:0', 'cuda:1'")

    # Data
    parser.add_argument("--sample_size", type=int, default=5000,
                        help="Number of samples in synthetic data")

    # Training
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--generator_batch_size", type=int, default=64)
    parser.add_argument("--generator_lr", type=float, default=1e-2)

    parser.add_argument("--outcome_lr", type=float, default=3e-2)
    parser.add_argument("--treatment_lr", type=float, default=3e-2)

    # parser.add_argument("--outcome_lr", type=float, default=1e-8)
    # parser.add_argument("--treatment_lr", type=float, default=1e-8)

    # Data generation constants
    parser.add_argument("--P0", type=float, default=20.47)
    parser.add_argument("--P1", type=float, default=33.75)

    # Monte Carlo
    parser.add_argument("--J", type=int, default=100)
    parser.add_argument("--M", type=int, default=5)
    parser.add_argument("--eps_dim", type=int, default=1)
    parser.add_argument("--u_dim", type=int, default=1)
    parser.add_argument("--eps_lambda", type=float, default=1)
    parser.add_argument("--ae_t_lambda_outcome", type=float, default=1e-3)
    parser.add_argument("--ae_z_lambda_outcome", type=float, default=1e-3)
    parser.add_argument("--entropy_lambda_outcome", type=float, default=1e-3)
    parser.add_argument("--ae_t_lambda_treatment", type=float, default=1e-3)
    parser.add_argument("--ae_w_lambda_treatment", type=float, default=1e-3)
    parser.add_argument("--entropy_lambda_treatment", type=float, default=1e-3)
    parser.add_argument("--entropy_sigma", type=float, default=1.0)

    parser.add_argument("--logging", action="store_true", help="Enable logging or not", default=False)

    # Autoencoder flag
    parser.add_argument("--use_ae", action="store_true", help="Enable AE in bridges")
    parser.add_argument("--use_entropy", action="store_true", help="Enable Entropy in bridges")

    args = parser.parse_args()

    # Convert device string to torch.device
    args.device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    return args
