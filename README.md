# ReginalWFN
Regional Weather Forecasting AI Model Based on Nvidia's FourCastNet

This README explains of using ReginalWFN for developers.

If you are interested in using ReginalWFN for actual weather forecasting cases,
please see REAME in `forecast` directory.

## Training

### Data Preparation

Data for developments is located at `/lustre/storm/nwp501/proj-shared/grnydawn/RegionalWFN/data/4x4`.

### Training Configuration

model.yaml
train.yaml, infer.yaml, or finetune.yaml
system.yaml
user.yaml

### Running Model

#### Running using a Batch System


#### Running on Interactive nodes

```
MASTER=$(hostname) && srun -n 4 -- echo $MASTER
```

### Output Verification


## Inferring

### Data Preparation


### Running Model


### Forecast Verification


## Fine Tuninig
