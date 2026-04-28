## DC-XB-model


### Installation

Clone the repository and create the conda environment:

```bash
git clone https://github.com/haosheng-21/DC-XB-model.git
cd DC-XB-model
conda env create -f environment_DC-XB-model.yml
conda activate dc_xb_env
```
Note: Directly creating the environment from the `.yml` file may not succeed in all cases due to platform differences and package compatibility issues. If the installation fails, we recommend manually installing the dependencies one by one according to the package versions specified in `environment_DC-XB-model.yml`.



### Generate materials

To generate materials using the fine-tuned model

```
python scripts/generate_newtopo.py --model_path='/.../DC-XB-model/diffcsp_trained_model/mp_gen' --batch_size=<batch_size> --load_model_decoder='/.../DC-XB-model/model_decoder_3class_topo.pth'

```

To generate materials using the original model (Simply avoid loading '/.../DC-XB-model/model_decoder_3class_topo.pth'.)

```
python scripts/generate_newtopo.py --model_path='/.../DC-XB-model/diffcsp_trained_model/mp_gen' --batch_size=<batch_size>

```


### Reinforcement Fine-Tuning

To perform ReFT

```
python scripts/train_ddpo.py --model_path='/.../DC-XB-model/diffcsp_trained_model/mp_gen' --batch_size=<batch_size>

```


### Note

The path ="/.../DC-XB-model" in files .env, XBERT/data.py, XBERT/model_finetune.py, XBERT/predict.py, and XBERT/utils.py should be replaced with the location of your own folder.

### Acknowlegement

Our implementation is built upon the DiffCSP++ framework, and we gratefully acknowledge the authors for their elegant and well-designed codebase.