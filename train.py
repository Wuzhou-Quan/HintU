import mmengine, gsettings, torch, os, click, torch.optim as optim, datetime
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.visualization import ClearMLVisBackend
from mmengine.optim.scheduler import CosineAnnealingLR, LinearLR
from inad_toolbox.metrics import ISTDMetrics

# Project name for ClearML
project_name = gsettings.project_name


@click.command()
@click.option("-m", "--model_arch_name", prompt="Input model arch in deployments folder", type=str, required=True, help="Input the arch in deployments folder")
@click.option("-t", "--train_dataset_name", prompt="Input training dataset in deployments folder", type=str, required=True, help="Input the training dataset in deployments folder")
@click.option("-v", "--val_dataset_name", prompt="Input validation dataset in deployments folder", type=str, required=True, help="Input the validation dataset in deployments folder")
@click.option("-b", "--batch_size", type=int, default=4)
@click.option("--max_epoches", type=int, default=300)
@click.option("--ckpt_path", type=str, default=None)
@click.option("--with_aug", type=bool, default=True)
@click.option("--lr", type=float, default=1e-3)
@click.option("--dataset_size_wh", type=int, default=(256, 256), nargs=2)
@click.option("--save_cp_interval", type=int, default=100)
@click.option("--val_interval", type=int, default=1)
def start_one_training_process(
    model_arch_name: str,
    train_dataset_name: str,
    val_dataset_name: str,
    batch_size: int,
    max_epoches: int,
    ckpt_path: str,
    with_aug: bool,
    lr: float,
    dataset_size_wh: tuple,
    save_cp_interval: int,
    val_interval: int,
):
    # Dataset
    train_dataset_deployment_file_path = gsettings.deployment_path / "datasets" / f"{train_dataset_name}.py"
    train_dataset_cfg = Config.fromfile(
        os.fspath(train_dataset_deployment_file_path),
        lazy_import=False,
    )
    train_dataset_cfg.merge_from_dict(dict(dataset=dict(output_size_wh=list(dataset_size_wh))))
    train_dataset_cfg.dataset["use_augment"] = with_aug
    train_dataloader = torch.utils.data.DataLoader(
        mmengine.DATASETS.build(train_dataset_cfg.dataset), batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False, prefetch_factor=2, persistent_workers=True, pin_memory=True
    )
    val_dataset_deployment_file_path = gsettings.deployment_path / "datasets" / f"{val_dataset_name}.py"
    val_dataset_cfg = Config.fromfile(
        os.fspath(val_dataset_deployment_file_path),
        lazy_import=False,
    )
    val_dataset_cfg.merge_from_dict(dict(dataset=dict(output_size_wh=list(dataset_size_wh))))
    val_dataloader = torch.utils.data.DataLoader(
        mmengine.DATASETS.build(val_dataset_cfg.dataset), batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False, prefetch_factor=2, persistent_workers=True, pin_memory=True
    )

    # Model
    model_deployment_file_path = gsettings.deployment_path / "models" / f"{model_arch_name}.py"
    model_cfg = Config.fromfile(
        os.fspath(model_deployment_file_path),
        lazy_import=False,
    )
    model: torch.nn.Module = mmengine.MODELS.build(model_cfg.model)

    # Optimizer Settings
    optim_wrapper = dict(optimizer=dict(type=optim.AdamW, lr=lr))
    param_scheduler = [dict(type=LinearLR, start_factor=1, end_factor=1e-1, by_epoch=False, begin=0, end=500), dict(type=CosineAnnealingLR, by_epoch=False, eta_min=0, begin=500)]

    runner = Runner(
        model=model,
        experiment_name=f"{model_arch_name}_{train_dataset_name}{'_noaug' if not with_aug else ''}_{datetime.datetime.now().strftime(r'%y%m%d_%H%M%S')}",
        work_dir=os.fspath(gsettings.exps_workdir_path / f"{model_arch_name}_{train_dataset_name}{'_noaug' if not with_aug else ''}"),
        train_dataloader=train_dataloader,
        optim_wrapper=optim_wrapper,
        param_scheduler=param_scheduler,
        train_cfg=dict(by_epoch=True, max_epochs=max_epoches, val_interval=val_interval),
        default_hooks=dict(checkpoint=dict(type="CheckpointHook", interval=save_cp_interval)),
        resume=False,
        val_evaluator=[
            dict(type=ISTDMetrics),
        ],
        val_dataloader=val_dataloader,
        val_cfg=dict(),
        load_from=ckpt_path,
        visualizer=dict(
            type="Visualizer",
            vis_backends=[
                dict(
                    type=ClearMLVisBackend,
                    init_kwargs=dict(
                        task_name=f"{model_arch_name}_{train_dataset_name}{'_noaug' if not with_aug else ''}_{datetime.datetime.now().strftime(r'%y%m%d_%H%M%S')}",
                        project_name=project_name,
                    ),
                )
            ],
        ),
    )
    runner.visualizer.add_config(config=model_cfg)
    runner.train()


if __name__ == "__main__":
    start_one_training_process()
