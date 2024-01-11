## Synthesize Boundaries: A Boundary-aware Self-consistent Framework for Weakly Supervised Salient Object Detection

## Usage
### 1. Download
##### 1.1 Download the DUTS and other datasets and unzip them into framework/data folder. Scribble labels can be downloaed from [Scribble Saliency](https://github.com/JingZhang617/Scribble_Saliency).
##### 1.2 Download ResNet-50 pretrained models and save it into framework folder ([Res50](https://drive.google.com/file/d/1arzcXccUPW1QpvBrAaaBv1CapviBQAJL/view))


### 2. Synthetic Image Generation
```bash
python synthetic_image_generation/main.py
```
save it into framework/data folder

### 3. Train 
```bash
python framework/train.py
```

### 4. Predict
Our saliency detector is SCWSSOD, so the saliency maps can be generated by combining the trained model parameters with the prediction code of SCWSSOD. Please refer to [SCWSSOD](https://github.com/siyueyu/SCWSSOD).

Pretrained models can be downloaded from [Google Drive](https://drive.google.com/file/d/1RNpQ8P6qug1TNGMdsrNZcHL1UIu73mWw/view?usp=share_link).

We provide the pre-computed saliency maps from our paper [Google Drive](https://drive.google.com/file/d/13CXJXPHg8ZHNDycmf4FMR9mMjwv71k6X/view?usp=drive_link).


##### Thanks to [SCWSSOD](https://github.com/siyueyu/SCWSSOD).
