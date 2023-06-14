# RegGeoNet: Learning Regular Representations for Large-Scale 3D Point Clouds

This is the official implementation of **[[RegGeoNet](https://link.springer.com/article/10.1007/s11263-022-01682-w)] (IJCV 2022)**, an unsupervised neural architecture to parameterize an unstructured 3D point cloud into a regular 2D image representation structure called deep geometry image (**DeepGI**), such that spatial coordinates of unordered 3D points are encoded in three-channel grid pixels.

<p align="center"> <img src="https://github.com/keeganhk/RegGeoNet/blob/master/imgs/toy_example.png" width="85%"> </p>

### Setup
This code has been tested with Python 3.9, PyTorch 1.10.1, CUDA 11.1 and cuDNN 8.0.5 on Ubuntu 20.04.

- Install the differentiable computer vision library [Kornia](https://kornia.readthedocs.io/en/latest/)
```
pip install git+https://github.com/kornia/kornia
```

### From Point Clouds to DeepGIs
Run ```para_pc.py``` under the ```scripts/para/``` folder to convert a given 3D point cloud into its DeepGI representation.

### Downstream Applications
Different downstream tasks can be performed directly on the generated DeepGIs, as an equivalent way of processing point cloud data. The pre-processed DeepGI-format datasets can be downloaded [here](https://drive.google.com/file/d/1TJxmi_xLVLYaV9eLCP57peS8TkLNpNQd/view?usp=sharing). Please put them under the ```data``` folder. Our pre-trained models are provided under the ```ckpt``` folder.

### Citation
If you find our work useful in your research, please consider citing:

	@article{zhang2022reggeonet,
	  title={RegGeoNet: Learning Regular Representations for Large-Scale 3D Point Clouds},
	  author={Zhang, Qijian and Hou, Junhui and Qian, Yue and Chan, Antoni B and Zhang, Juyong and He, Ying},
	  journal={International Journal of Computer Vision},
      volume={130},
      number={12},
      pages={3100--3122},
	  year={2022}
	}
  
  
