from colbert.infra.config.settings import IndexingSettings, ResourceSettings


def copy_essential_config(source_config, target_config):
    for field in IndexingSettings.__dataclass_fields__:
        setattr(target_config, field, getattr(source_config, field))

    for field in ResourceSettings.__dataclass_fields__:
        setattr(target_config, field, getattr(source_config, field))

    target_config.experiment = source_config.experiment
    target_config.ignore_scores = source_config.ignore_scores
