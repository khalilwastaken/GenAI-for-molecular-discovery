try:
    import graph_tool as gt
except ImportError:
    gt = None
    print("graph_tool not installed, continuing without it")
import os
import pathlib
import warnings

import torch
torch.cuda.empty_cache()
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from src import utils
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics

from diffusion_model import LiftedDenoisingDiffusion
from diffusion_model_discrete import DiscreteDenoisingDiffusion
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures

warnings.filterwarnings("ignore", category=PossibleUserWarning)

def get_resume(cfg, model_kwargs):
    saved_cfg = cfg.copy()
    name = cfg.general.name + '_resume'
    resume = cfg.general.test_only
    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)
    cfg = model.cfg
    cfg.general.test_only = resume
    cfg.general.name = name
    cfg = utils.update_config_with_new_keys(cfg, saved_cfg)
    return cfg, model

def get_resume_adaptive(cfg, model_kwargs):
    saved_cfg = cfg.copy()
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = current_path.split('outputs')[0]
    resume_path = os.path.join(root_dir, cfg.general.resume)

    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume_path, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion.load_from_checkpoint(resume_path, **model_kwargs)
    new_cfg = model.cfg
    for category in cfg:
        for arg in cfg[category]:
            new_cfg[category][arg] = cfg[category][arg]
    new_cfg.general.resume = resume_path
    new_cfg.general.name = new_cfg.general.name + '_resume'
    new_cfg = utils.update_config_with_new_keys(new_cfg, saved_cfg)
    return new_cfg, model

@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    # ==============================================================================
    # 1. VERROUILLAGE DU DATASET ET CORRECTION OMEGACONF
    # ==============================================================================
    OmegaConf.set_struct(cfg, False) # Déverrouillage de la configuration
    
    # On garantit que le dataset utilisé est bien 'odor' pour la GenAI for molecular discovery
    assert cfg.dataset.name == 'odor', f"ERREUR: Dataset détecté = {cfg.dataset.name}. Veuillez lancer avec dataset=odor"

    if not cfg.train.save_model:
        print("ATTENTION: save_model était False, on le force à True.")
        cfg.train.save_model = True
        
    OmegaConf.set_struct(cfg, True) # Reverrouillage de sécurité
    # ==============================================================================

    dataset_config = cfg["dataset"]

    if dataset_config["name"] in ['sbm', 'comm20', 'planar']:
        from datasets.spectre_dataset import SpectreGraphDataModule, SpectreDatasetInfos
        from analysis.spectre_utils import PlanarSamplingMetrics, SBMSamplingMetrics, Comm20SamplingMetrics
        from analysis.visualization import NonMolecularVisualization

        datamodule = SpectreGraphDataModule(cfg)
        if dataset_config['name'] == 'sbm':
            sampling_metrics = SBMSamplingMetrics(datamodule)
        elif dataset_config['name'] == 'comm20':
            sampling_metrics = Comm20SamplingMetrics(datamodule)
        else:
            sampling_metrics = PlanarSamplingMetrics(datamodule)

        dataset_infos = SpectreDatasetInfos(datamodule, dataset_config)
        train_metrics = TrainAbstractMetricsDiscrete() if cfg.model.type == 'discrete' else TrainAbstractMetrics()
        visualization_tools = NonMolecularVisualization()

        if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)
        
        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}

    elif dataset_config["name"] in ['qm9', 'guacamol', 'moses', 'odor']:
        from metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics
        from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
        from diffusion.extra_features_molecular import ExtraMolecularFeatures
        from analysis.visualization import MolecularVisualization

        if dataset_config["name"] == 'qm9':
            from datasets import qm9_dataset
            datamodule = qm9_dataset.QM9DataModule(cfg)
            dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
            train_smiles = qm9_dataset.get_train_smiles(cfg=cfg, train_dataloader=datamodule.train_dataloader(),
                                                        dataset_infos=dataset_infos, evaluate_dataset=False)
        elif dataset_config['name'] == 'guacamol':
            from datasets import guacamol_dataset
            datamodule = guacamol_dataset.GuacamolDataModule(cfg)
            dataset_infos = guacamol_dataset.Guacamolinfos(datamodule, cfg)
            train_smiles = None
        elif dataset_config.name == 'moses':
            from datasets import moses_dataset
            datamodule = moses_dataset.MosesDataModule(cfg)
            dataset_infos = moses_dataset.MOSESinfos(datamodule, cfg)
            train_smiles = None
        elif dataset_config["name"] == 'odor':
            from datasets import odor_dataset
            datamodule = odor_dataset.OdorDataModule(cfg)
            dataset_infos = odor_dataset.OdorInfos(datamodule=datamodule, cfg=cfg)
            train_smiles = None
        else:
            raise ValueError("Dataset not implemented")

        if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
            domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
            domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)

        if cfg.model.type == 'discrete':
            train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
        else:
            train_metrics = TrainMolecularMetrics(dataset_infos)

        sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)
        visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)

        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}
    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

    if cfg.general.test_only:
        cfg, _ = get_resume(cfg, model_kwargs)
        os.chdir(cfg.general.test_only.split('checkpoints')[0])
    elif cfg.general.resume is not None:
        cfg, _ = get_resume_adaptive(cfg, model_kwargs)
        os.chdir(cfg.general.resume.split('checkpoints')[0])

    utils.create_folders(cfg)

    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion(cfg=cfg, **model_kwargs)

    callbacks = []
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                              filename='{epoch}',
                                              monitor='val/epoch_NLL',
                                              save_top_k=5,
                                              mode='min',
                                              every_n_epochs=1)
        last_ckpt_save = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}", filename='last', every_n_epochs=1)
        callbacks.append(last_ckpt_save)
        callbacks.append(checkpoint_callback)

    if cfg.train.ema_decay > 0:
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks.append(ema_callback)

    name = cfg.general.name
    use_gpu = torch.cuda.is_available() and int(getattr(cfg.general, "gpus", 0)) > 0
    print("torch.cuda.is_available() =", torch.cuda.is_available())
    print("use_gpu =", use_gpu)

    # ==============================================================================
    # 2. FIX CRITIQUE PYTORCH LIGHTNING : COUPER LA VALIDATION
    # ==============================================================================
    trainer = Trainer(
        gradient_clip_val=cfg.train.clip_grad,
        strategy="ddp_find_unused_parameters_true",
        accelerator="gpu" if use_gpu else "cpu",
        devices=cfg.general.gpus if use_gpu else 1,
        max_epochs=cfg.train.n_epochs,
        check_val_every_n_epoch=1000,    # Empêche la validation pendant l'entraînement
        num_sanity_val_steps=0,          # Empêche le sanity check au démarrage
        fast_dev_run=cfg.general.name == 'debug',
        enable_progress_bar=False,
        callbacks=callbacks,
        log_every_n_steps=50 if name != 'debug' else 1,
        logger=[]
    )
    # ==============================================================================

    if not cfg.general.test_only:
        # Entraînement / Reprise
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
        
        # ==============================================================================
        # 3. SAUVEGARDE DE SÉCURITÉ DU MODÈLE
        # ==============================================================================
        print("\n" + "="*50)
        print("SAUVEGARDE DE SÉCURITÉ DU MODÈLE")
        root_ckpt_path = os.path.join(os.getcwd(), "final_model_manual.ckpt")
        trainer.save_checkpoint(root_ckpt_path)
        print(f"Modèle sauvegardé manuellement ici : {root_ckpt_path}")
        print("Utilisez ce chemin pour generate_odor.py !")
        print("="*50 + "\n")
        # ==============================================================================

        # Test (avec protection anti-crash)
        if cfg.general.name not in ['debug', 'test']:
            try:
                trainer.test(model, datamodule=datamodule)
            except Exception as e:
                print(f"\n[WARNING] Le test a échoué (erreur: {e}), mais le modèle est SAUVÉ.")
                print(f"Vous pouvez utiliser {root_ckpt_path}")

    else:
        trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only)

if __name__ == '__main__':
    main()