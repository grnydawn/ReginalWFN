# cfg directory
This directory includes model configurations for developing ReginalWFN.

The yaml files in this directory are read in the order of:

model.yaml -> one of train.yaml, infer.yaml, or finetune.yaml ->
system.yaml -> user.yaml -> case.yaml

The latter yaml file overrides previous configurations.

