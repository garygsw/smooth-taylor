# Smooth-Taylor

_SmoothTaylor_ is a gradient-based attribution method derived from the _Taylor's theorem_ for **deep neural network attribution**. It is proposed as a theoretical bridge between _SmoothGrad_ [(Smilkov et al.)](https://arxiv.org/abs/2004.10484) and _Integrated Gradients_ [(Sundararajan et al.)](https://arxiv.org/abs/1703.01365).

![sample](method_comparison.png "Sample saliency maps of different attribution methods")

In our paper, we conduct experiments to compare the performance of _SmoothTaylor_ and _Integrated Gradients_ using empirical quantitative measures: _perturbations scores_ and _average total variation_, and show that _SmoothTaylor_ is able to generate attribution maps that are smoother and more sensitive.

This repository includes a **PyTorch** implementation of _SmoothTaylor_, _SmoothGrad_ and _Integrated Gradients_.

## Paper

Goh, S. W. Goh, S. Lapuschkin, L. Weber, W. Samek, and A. Binder (2021). “Un- derstanding Integrated Gradients with SmoothTaylor for Deep Neural Network Attribution”. In: 2020 25th International Conference on Pattern Recognition (ICPR), pp. 4949–4956. DOI:10.1109/ICPR48806.2021.9413242.

Links: [Paper](https://arxiv.org/abs/2004.10484) • [Code](https://github.com/garygsw/smooth-taylor) • [Presentation](https://www.dropbox.com/s/xjb1xw6ynlwb7xa/ICPR_2020_Paper_Presentation.pdf?dl=0) • [Poster](https://www.dropbox.com/s/eks6ajkjejyf6tc/Poster%201363%20Understanding%20Integrated%20Gradients%20with%20SmoothTaylor%20for%20Deep%20Neural%20Network%20Attribution.pdf?dl=0)

## Setup

### Requirement

Required Python (version 3.7) with standard libraries and following packages version requirements (tested for execution):

-   pytorch 1.4.0
-   torchvision 0.5.0
-   scikit-image 0.16.2
-   pillow 7.0.0
-   numpy 1.7.14
-   scipy 1.4.1
-   tqdm 4.36.1

Tested in Ubuntu + Intel i7-6700 CPU + RTX 2080 Ti with Cuda (10.1). CPU-only mode also possible, but running with GPU is highly recommended.

### Dataset

The first 1000 images of the [ILSVRC2012 ImageNet object recognition](http://www.image-net.org/challenges/LSVRC/2012/) **validation** dataset is used in our paper's experiment. To replicate our experiment using our experiment code, [download](http://academictorrents.com/details/5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5) or place the dataset into a new folder `./data`, and put the annotations file (in `.xml` formats) in subfolder `./data/annotations` and the raw images in subfolder `./data/images`. Note: required resource files and pre-processing steps for ImageNet are already provided in `./rsc` and in `./attribution/constants.py`.

```
# the ILSVRC2012 ImageNet validation dataset structure should be placed like this
    data/
        -annotations/
            -ILSVRC2012_val_{img#}.xml
        -images/
            -ILSVRC2012_val_{img#}.JPEG
```

You may also create your own dataset using the PyTorch's `torch.utils.data.Dataset` wrapper to use in your own code.

### Models

In our experiment, we applied attribution on the following deep neural image classifiers:

-   DenseNet121 [(Huang et al., 2017)](https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py), and;
-   ResNet152 [(He et al., 2015)](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py).

They are both pretrained on the ILSVRC2012 ImageNet dataset, and we use the instance in the default `torchvision` url paths. You may use other pretrained image classifier models that are implemented in PyTorch. Just remember to add the name and instance in the `MODELS` dictionary map in `./attribution/constants.py`.

```python
# the current default mapping is:
from torchvision import models
MODELS = {'densenet121': models.densenet121,
          'resnet152': models.resnet152}
```

## Execution

To replicate our experiments, please follow the steps in this section.

1. First, save the classification outputs of the images using the pre-trained image classifiers:

    ```bash
    python experiment_classify.py [-m MODEL_NAME] [-b BATCH_SIZE]
    ```

    Arguments:

    - `-m MODEL_NAME`: use `densenet121` or `resnet152`
    - `-b BATCH_SIZE` (optional): number of image per epoch (default: 128)

    The classification output should be saved in a new folder `./output/[MODEL_NAME]`.

1. Perform the neural network attribution. We implemented 3 gradient-based attribution methods here:

    1. _SmoothTaylor_

        ```bash
        python experiment_smooth_taylor.py [-m MODEL_NAME] [-b BATCH_SIZE]
                                           [-s NOISE_SCALE] [-r NUM_ROOTS]
        ```

        Arguments:

        - `-m MODEL_NAME`: use `densenet121` or `resnet152`
        - `-b BATCH_SIZE` (optional): number of image per epoch (default: 50)
        - `-s NOISE_SCALE` (optional): magnitude of the noise scale to noise the image (default: 5e-1)
        - `-n NUM_ROOTS` (optional): number of noise inputs to use (default: 150)

    1. _IntegratedGradients_

        ```bash
        python experiment_ig.py [-m MODEL_NAME] [-b BATCH_SIZE] [-k STEPS]
                                [-z BASELINE_TYPE] [-n NUM_NOISE]
        ```

        Arguments:

        - `-m MODEL_NAME`: use `densenet121` or `resnet152`
        - `-b BATCH_SIZE` (optional): number of image per epoch (default: 50)
        - `-k STEPS` (optional): number of steps along path (default: 50)
        - `-z BASELINE_TYPE` (optional): baseline type [use `zero` or `noise`] (default: `zero`)
        - `-n NUM_NOISE` (optional): number of noise baselines to use (default: 1)

    1. _SmoothGrad_

        ```bash
        python experiment_grad.py [-m MODEL_NAME] [-b BATCH_SIZE] [-s] [-p NOISE_SCALE] [-n NUM_NOISE]
        ```

        Arguments:

        - `-m MODEL_NAME`: use `densenet121` or `resnet152`
        - `-b BATCH_SIZE` (optional): number of image per epoch (default: 50)
        - `-s` (optional): to use SmoothGrad or not (default: `False`)
        - `-p NOISE_SCALE` (optional): percentage noise scale (default: 15)
        - `-n NUM_NOISE` (optional): number of noise inputs to use (default: 50)

    The heatmaps should be saved in a new folder `./heatmaps`, with hyperparameter values as subfolders names e.g. `./heatmaps/[ATTRIBUTION_METHOD]/[MODEL_NAME]/...`

1. Evaluate the attribution methods by comparing their heatmaps, using two quantitative evaluation metrics:

    1. _Perturbation Scores_ for sensitivity

        ```bash
        python experiment_perturbations.py [-m MODEL_NAME] [-a ANALYZER] [-b BATCH_SIZE]
                                           [-z BASELINE] [-n NUM_NOISE]
                                           [-s NOISE_SCALE] [-r NUM_ROOTS]
                                           [-k KERNEL_SIZE] [-pt NUM_PERTURBS]
                                           [-l NUM_REGIONS] [-an]
                                           [-af ADAPTIVE_FUNCTION]
        ```

        Arguments:

        - `-m MODEL_NAME`: use `densenet121` or `resnet152`
        - `-a ANALYZER`: attribution method [use `grad`, `smooth-grad`, `smooth-taylor`, or `ig`]
        - `-b BATCH_SIZE` (optional): number of image per epoch (default: 50)
        - `-z BASELINE` (optional): IG baseline used [use `zero` or `noise`] (default: `zero`)
        - `-n NUM_NOISE` (optional): number of noised baseline in IG (default: 1)
        - `-s NOISE_SCALE` (optional): magnitude of noise scale for smoothing (default: 5e-1)
        - `-r NUM_ROOTS` (optional): number of noise inputs for smoothing (default: 150)
        - `-k KERNEL_SIZE` (optional): size of the window of each perturbation (default: 15)
        - `-pt NUM_PERTURBS` (optional): number of random perturbations to evaluate (default: 50)
        - `-l NUM_REGIONS` (optional): number of regions to perturbate (default: 30)
        - `-an` (optional): use adaptive noise (default: `False`)
        - `-af ADAPTIVE_FUNCTION` (optional): objective function for adaptive noising [use `aupc` or `autvc`] (default: `aupc`)

    1. _Average Total Variation_ for noisiness

        ```bash
        python experiment_total_variation.py [-m MODEL_NAME] [-a ANALYZER]
                                             [-z BASELINE] [-n NUM_NOISE]
                                             [-s NOISE_SCALE] [-r NUM_ROOTS]
                                             [-ds DOWNSCALE] [-wms WIDTH_MIN_SIZE]
                                             [-hms HEIGHT_MIN_SIZE] [-lp LP_NORM]
                                             [-an] [-af ADAPTIVE_FUNCTION]
        ```

        Arguments:

        - `-m MODEL_NAME`: use `densenet121` or `resnet152`
        - `-a ANALYZER`: attribution method [use `grad`, `smooth-grad`, `smooth-taylor`, or `ig`]
        - `-z BASELINE` (optional): IG baseline used [use `zero` or `noise`] (default: zero)
        - `-n NUM_NOISE` (optional): number of noised baseline in IG (default: 1)
        - `-s NOISE_SCALE` (optional): magnitude of noise scale for smoothing (default: 5e-1)
        - `-r NUM_ROOTS` (optional): number of noise inputs for smoothing (default: 150)
        - `-ds DOWNSCALE` (optional): factor to downscale heatmap (default: 1.5)
        - `-wms WIDTH_MIN_SIZE` (optional): minimum width for downscale (default: 30)
        - `-hms HEIGHT_MIN_SIZE` (optional): minimum height for downscale (default: 30)
        - `-lp LP_NORM` (optional): norm to use to calculate total variation (default: 1)
        - `-an` (optional): use adaptive noise (default: `False`)
        - `-af ADAPTIVE_FUNCTION` (optional): objective function for adaptive noising [use `aupc` or `autvc`] (default: `aupc`)

1. Generate _SmoothTaylor_ heatmaps with _adaptive noising_ hyperparameter tuning technique:

    ```bash
    python experiment_adaptive_noising.py [-m MODEL_NAME] [-b BATCH_SIZE]
                                          [-r NUM_ROOTS] [-f OBJ_FUNCTION]
                                          [-ds DOWNSCALE] [-wms WIDTH_MIN_SIZE]
                                          [-hms HEIGHT_MIN_SIZE] [-lp LP_NORM]
                                          [-k KERNEL_SIZE] [-p NUM_PERTURBS]
                                          [-l NUM_REGIONS] [-lr LEARNING_RATE]
                                          [-y LEARNING_DECAY] [-c MAX_STOP_COUNT]
                                          [-x MAX_ITERATION]
    ```

    Arguments:

    - `-m MODEL_NAME`: use `densenet121` or `resnet152`
    - `-b BATCH_SIZE` (optional): number of image per epoch (default: 50)
    - `-r NUM_ROOTS` (optional): number of noise inputs for smoothing (default: 150)
    - `-f OBJ_FUNCTION` (optional): objective function for adaptive noising [use `aupc` or `autvc`] (default: `aupc`)
    - `-ds DOWNSCALE` (optional): factor to downscale heatmap (default: 1.5)
    - `-wms WIDTH_MIN_SIZE` (optional): minimum width for downscale (default: 30)
    - `-hms HEIGHT_MIN_SIZE` (optional): minimum height for downscale (default: 30)
    - `-lp LP_NORM` (optional): norm to use to calculate total variation (default: 1)
    - `-k KERNEL_SIZE` (optional): size of the window of each perturbation (default: 15)
    - `-p NUM_PERTURBS` (optional): number of random perturbations to evaluate (default: 50)
    - `-l NUM_REGIONS` (optional): number of regions to perturbate (default: 30)
    - `-lr LEARNING_RATE` (optional): learning rate for variable update (default: 0.1)
    - `-y LEARNING_DECAY` (optional): decay rate of learning rate (default: 0.9)
    - `-c MAX_STOP_COUNT` (optional): maximum stop count to terminate search (default: 3)
    - `-x MAX_ITERATION` (optional): maximum iterations to search (default: 20)

    Perform evaluation (see Step 2 above) if required.

For clearer explanations to what each hyperparameter in the arguments mean, please refer to our paper.

## License

This work is licensed under MIT License. See [LICENSE](LICENSE.md) for details.

If you find our code or paper useful, please cite our paper:

```
@inproceedings{goh2020understanding,
  author    = {Gary S. W. Goh and
               Sebastian Lapuschkin and
               Leander Weber and
               Wojciech Samek and
               Alexander Binder},
  title     = {Understanding Integrated Gradients with SmoothTaylor for Deep Neural
               Network Attribution},
  booktitle = {25th International Conference on Pattern Recognition, {ICPR} 2020,
               Virtual Event / Milan, Italy, January 10-15, 2021},
  pages     = {4949--4956},
  publisher = {{IEEE}},
  year      = {2020},
  doi       = {10.1109/ICPR48806.2021.9413242},
}
```

## Questions

If you found any bugs, or have any questions, please email to garygsw@gmail.com.
