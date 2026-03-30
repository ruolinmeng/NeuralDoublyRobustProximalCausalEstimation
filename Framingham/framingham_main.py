from framingham_config import *
from framingham_data import generate_framingham_data, make_tensors
from framingham_utils import set_seed
from framingham_train import train_outcome_model, train_treatment_model
from framingham_evaluate import evaluate_models

def run_experiment(seed, cfg):
    set_seed(seed)
    train_df, val_df = generate_framingham_data()
    W_t_train, Z_t_train, A_t_train, Y_t_train, W_all_train, Z_all_train = make_tensors(train_df, cfg.device)
    W_t_val, Z_t_val, A_t_val, Y_t_val, W_all_val, Z_all_val = make_tensors(val_df, cfg.device)

    # outcome model training
    set_seed(seed)
    best_outcome_models, best_val_outcome = train_outcome_model(
        cfg, W_t_train, Z_t_train, A_t_train, Y_t_train,
        W_all_train, Z_all_train,
        W_t_val, Z_t_val, A_t_val, Y_t_val,
        W_all_val, Z_all_val,
        seed=seed, USE_AE=cfg.use_ae, USE_ENTROPY=cfg.use_entropy
    )

    # treatment model training
    set_seed(seed)
    best_treatment_models, best_val_treatment = train_treatment_model(
        cfg, W_t_train, Z_t_train, A_t_train, Y_t_train,
        W_all_train, Z_all_train,
        W_t_val, Z_t_val, A_t_val, Y_t_val,
        W_all_val, Z_all_val,
        seed=seed, USE_AE=cfg.use_ae, USE_ENTROPY=cfg.use_entropy
    )

    # evaluation of the models
    set_seed(seed)
    evaluate_models(
        seed, cfg, W_t_train, Z_t_train, A_t_train, Y_t_train,
        best_outcome_models, best_treatment_models,
        results_dir=f"results/{cfg.run_name}/seed_{seed}"
    )

    return best_val_outcome, best_val_treatment

def main():
    cfg = get_config()
    print("Running:", cfg.run_name, "on", cfg.device,
          "with USE_AE =", cfg.use_ae, "with USE_ENTROPY =", cfg.use_entropy)

    for seed in cfg.seeds:
        print(f"Running experiment with seed {seed}...")
        best_val_outcome, best_val_treatment = run_experiment(seed, cfg)
        print(f"Seed {seed} → Val Outcome={best_val_outcome:.4f}, Val Treatment={best_val_treatment:.4f}")

if __name__ == "__main__":
    main()
