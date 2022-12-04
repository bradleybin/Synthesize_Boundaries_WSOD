 
### Synthesize Boundaries: A Boundary-aware Self-consistent Framework for Weakly Supervised Salient Object Detection


This is the code of our paper.  We are very willing to upload the trained parameters, unfortunately, we cannot due to the limitation of the size of supplementary materials. But we provide all codes. Furthermore, we upload the **final predicted saliency maps** on five datasets and  **synthetic images** on DUT-TE dataset.
.
## Usage
#### 1.Download
1.1 Download the ` DUTS`  and other datasets and unzip them into `code/framework/data` folder.

1.2 Download ResNet-50 pretrained models and save it into `code/framework` folder

#### 2.Synthetic Image Generation
```
python code/synthetic_image_generation/main.py
```
save it into `code/framework/data` folder

#### 3.Train Self-consistent Framework
```
python code/framework/train.py
```