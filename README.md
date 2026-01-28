## DC-XB-model


### Generate materials

To generate materials using the fine-tuned model

```
python scripts/generate_newtopo.py --model_path='/.../DC-XB-model/diffcsp_trained_model/mp_gen' --batch_size=<batch_size> --load_model_decoder='/.../DC-XB-model/model_decoder_3class_topo.pth'

```

To generate materials using the original model (Simply avoid loading '/.../DC-XB-model/model_decoder_3class_topo.pth'.)

```
python scripts/generate_newtopo.py --model_path='/.../DC-XB-model/diffcsp_trained_model/mp_gen' --batch_size=<batch_size>

```

### Note

The path ="/.../DC-XB-model" in files .env, XBERT/data.py, XBERT/model_finetune.py, XBERT/predict.py, and XBERT/utils.py should be replaced with the location of your own folder.

### Acknowlegement

Our implementation is built upon the DiffCSP++ framework, and we gratefully acknowledge the authors for their elegant and well-designed codebase.