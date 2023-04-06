import hydra
from omegaconf import DictConfig, OmegaConf

from memory_re.settings import CONFIGS_DIR, STORAGE_DIR, WANDB_PROJECT_NAME

OmegaConf.register_new_resolver("storage_dir", lambda: STORAGE_DIR)
OmegaConf.register_new_resolver("wandb_project_name", lambda: WANDB_PROJECT_NAME)


@hydra.main(version_base=None, config_path=str(CONFIGS_DIR))
def run(cfg: DictConfig) -> float:
    print(OmegaConf.to_yaml(cfg))
    runner = hydra.utils.get_method(cfg['runner']['_target_'])
    return runner(cfg)


if __name__ == '__main__':
    run()
