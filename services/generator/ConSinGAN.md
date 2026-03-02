# Objective
Since 2022 an interest for learning generative models from a single image, as oppossed to from a large dataset has been grown. This service has the task of generating images when a __new defect__ has been detected, to simply to augment the training data for YOLO.

__Concurrent Single GAN__ or ConSinGAN is an architecture that instead of producing images produce feature maps at each stage, training multiple stages concurrently, they modify the pyramid of generators $\{G_0,\dots,G_N\}$ that SinGAN used adapting the rescaling to not be strictly geometric.  


Reference [Improved techniques for training simple-image GANS](https://www.tobiashinz.com/2020/03/24/improved-techniques-for-training-single-image-gans.html).