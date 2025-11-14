# 🤖 Diffusion-Based Whole-body Visuomotor Learning

This project implements a diffusion-based visuomotor policy for whole-body robotic control. Our work builds upon the [**RoboPanoptes**](https://github.com/real-stanford/robopanoptes) system to replicate and adapt their model architecture and training pipeline. Demonstration data was collected using the open-source [**LeRobot-DIY**](https://github.com/yucy207/lerobot_diy) teleoperation system.

## 📌 Overview

- ACT, Diffusion or PI-0 can be used in our robot !!!
- ✅ Reproduced the RoboPanoptes diffusion policy using our own dataset  
- 🎮 Collected demonstrations using LeRobot-DIY with custom robot hardware  
- 🧠 Trained a Transformer-based diffusion policy using multi-view RGB and proprioceptive inputs  
- 🔧 Deployed learned policy for real-world sweeping and manipulation tasks  

<img width="100%" src="teaser.jpg">

## 📂 Project Structure

```
MyRobotProject/
├── configs/                         # Hydra configs for training
├── dataset.zarr.zip                 # Processed demonstration dataset
├── train.py                         # Training script
├── eval_real.py                     # Evaluation on real robot
├── teleop/
│   └── run_env.py                   # Teleop data collection script
├── process_data.py                  # Convert raw demos to zarr format
├── requirements.txt                 # Python dependencies
└── README.md
```

## 🧱 Installation

```bash
git clone https://github.com/<your-username>/<your-project>.git
cd <your-project>
pip install -r requirements.txt
```

> This project reuses code and configuration from [RoboPanoptes](https://github.com/real-stanford/robopanoptes).

## 📹 Data Collection (via LeRobot-DIY)

We used [**LeRobot-DIY**](https://github.com/yucy207/lerobot_diy) for teleoperation and demonstration recording.

### 1. Identify motor ports

```bash
ls /dev/serial/by-id
```

Update `client_port` and `leader_port` in `teleop/run_env.py`.

### 2. Identify camera device paths

```bash
ls /dev/v4l/by-path
```

Set `device_ids` in `teleop/run_env.py`.

### 3. Collect demonstrations

```bash
python teleop/run_env.py
```

### 4. Process raw data

```bash
python process_data.py
```

This will generate `dataset.zarr.zip` for training.

## 🧠 Training the Policy

We follow the training setup from RoboPanoptes.

### Single-GPU Training

```bash
python train.py --config-name=train_diffusion_transformer_snake_workspace task.dataset_path=dataset.zarr.zip
```

### Multi-GPU Training

```bash
accelerate launch --num_processes <num_gpus> train.py --config-name=train_diffusion_transformer_snake_workspace task.dataset_path=dataset.zarr.zip
```

## 🧪 Real-World Evaluation

### 1. Download pretrained checkpoint

```bash
wget https://real.stanford.edu/robopanoptes/data/pretrained_models/sweep.ckpt
```

### 2. Configure `eval_real.py`

- Set Dynamixel port paths  
- Set camera device IDs

### 3. Run evaluation

```bash
python eval_real.py
```

## 🙏 Acknowledgements

This project is built upon the following:

- **RoboPanoptes**  
  [Project Page](https://robopanoptes.github.io)  
  [Paper on arXiv](https://arxiv.org/abs/2501.05420)  
  ```bibtex
  @article{xu2025robopanoptes,
    title={RoboPanoptes: The All-seeing Robot with Whole-body Dexterity},
    author={Xu, Xiaomeng and Bauer, Dominik and Song, Shuran},
    journal={arXiv preprint arXiv:2501.05420},
    year={2025}
  }
  ```

- **LeRobot-DIY**  
  [GitHub Repository](https://github.com/yucy207/lerobot_diy)
  
- **Intro-Guide**  
  [Guide](https://deepwiki.com/real-stanford/RoboPanoptes)

- **Hugging_face**
  [Enter_datacollection](https://huggingface.co/yucy207)

## 📜 License

This project is released under the MIT License. See the [LICENSE](./LICENSE) file for more information.
