from sem_config import *
from sem_data import generate_sem_data, make_tensors
from sem_utils import set_seed
from sem_train import train_outcome_model, train_treatment_model
from sem_evaluate import evaluate_models
from sem_config import get_config

def run_experiment(seed, cfg):
    set_seed(seed)

    # --- Generate SEM data ---
    df = generate_sem_data(cfg.sample_size)
    W_t, Z_t, A_t, Y_t, X_t = make_tensors(df, cfg.device)

    # --- Train outcome bridge ---
    set_seed(seed)
    best_outcome_models = train_outcome_model(
        cfg, W_t, Z_t, A_t, Y_t, X_t,
        seed=seed, USE_AE=cfg.use_ae, USE_ENTROPY=cfg.use_entropy
    )

    # --- Train treatment bridge ---
    set_seed(seed)
    best_treatment_models = train_treatment_model(
        cfg, W_t, Z_t, A_t, Y_t, X_t,
        seed=seed, USE_AE=cfg.use_ae, USE_ENTROPY=cfg.use_entropy
    )

    # --- Evaluate models ---
    set_seed(seed)
    evaluate_models(
        seed, cfg, W_t, Z_t, A_t, Y_t, X_t,
        best_outcome_models, best_treatment_models,
        results_dir=f"results/{cfg.run_name}/seed_{seed}"
    )

def main():
    cfg = get_config()
    print(f"Running: {cfg.run_name} on {cfg.device}, USE_AE={cfg.use_ae}, USE_ENTROPY={cfg.use_entropy}")

    for seed in cfg.seeds:
        print(f"Running experiment with seed {seed}...")
        run_experiment(seed, cfg)

if __name__ == "__main__":
    main()
