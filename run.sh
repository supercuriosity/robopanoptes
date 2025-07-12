huggingface-cli download --repo-type dataset  yucy207/robopanoptes_test12 --cache-dir ./d

python process_data_from_lerobot.py 
python train.py --config-name=train_diffusion_transformer_snake_workspace task.dataset_path=./dataset.zarr.zip 