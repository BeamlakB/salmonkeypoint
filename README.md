## CNN-Based Key-point identification Models for Fish Length Estimation
This study presents a keypoint detection pipeline for localising nine anatomical landmarks on individual fish images
from the Fish Instance Benchmark (FIB) dataset. We evaluate two model variants: a lightweight Lite-HRNet architecture trained from random initialisation, and a full HRNet-W32 backbone pretrained on ImageNet-1K fine-tuned with a two-stage differential learning-rate strategy. The from-
scratch model failed to converge meaningfully on the 283-image training set, achieving an Object Keypoint Similarity (OKS) of only 0.008. Replacing it with the pretrained backbone increased OKS to 0.676, with per-keypoint mean errors of 1.0–3.1 pixels in heatmap space, demonstrating
that transfer learning is essential when labelled data is scarce. 

### Instruction on how to run the code 


```
Install dependencies (once):
    pip install timm torch torchvision opencv-python pillow numpy

Usage:
    Train (downloads HRNet-W32 weights automatically on first run)
    python hrnet_train.py
    python lite_train.py
     Resume from checkpoint
    python hrnet_train.py --checkpoint checkpoints/ckpt_epoch050.pth
    python lite_train.py --checkpoint checkpoints/ckpt_epoch050.pth

    # Evaluate on test set
    python  lite_train.py --mode test --checkpoint checkpoints/best_oks_model.pth
```
