# Inverse Graphics Energy Networks (IGE-Net)

Code for [IGE-Net: Inverse Graphics Energy Networks for 3D Human Pose Estimation and Object Reconstruction](http://openaccess.thecvf.com/content_CVPR_2019/papers/Jack_IGE-Net_Inverse_Graphics_Energy_Networks_for_Human_Pose_Estimation_and_CVPR_2019_paper.pdf) ([poster](https://jackd.github.io/images/ige-poster.pdf)).

```
@InProceedings{jack2019ige,
author = {Jack, Dominic and Maire, Frederic and Shirazi, Sareh and Eriksson, Anders},
title = {IGE-Net: Inverse Graphics Energy Networks for Human Pose Estimation and Single-View Reconstruction},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```

## Quick-start

Install [tensorflow>=1.13,<2.0](https://github.com/tensorflow.org).

```bash
git clone https://github.com/jackd/ige.git
cd ige
pip install -r requirements.txt
pip install -e .

# download data for human pose estimation
H3M_LIFT=~/tensorflow_datasets/downloads/manual/h3m_lift
mkdir -p $H3M_LIFT
wget -O https://www.dropbox.com/s/e35qv3n6zlkouki/h36m.zip $H3M_LIFT

# train tiny baseline model
python -m ige --gin_file=hpe/b128-1 --command=train --epochs=5
python -m ige --gin_file=hpe/b128-1 --command=evaluate --gin_param='include_procrustes=True'
# visualize the results
python -m ige --gin_file=hpe/b128-1 --command=vis

# train small IGE model
python -m ige --gin_file=hpe/ige4-b128-1 --command=train --epochs=5
python -m ige --gin_file=hpe/ige4-b128-1 --command=evaluate \
    --gin_param='include_procrustes=True'
```

## Project Structure

Each experiment is highly configurable, though is predominantly defined by 2 objects:

* a [Problem](https://github.com/jackd/ige/tree/master/ige/problem.py), specifying the data, loss, metrics etc.; and
* an `inference_fn` which maps inputs to predictions.

Implementations are in subdirectories (`hpe` (human pose estimation) and `vox` (single view object voxel reconstruction, coming soon)).

We make extensive use of [gin-config](https://github.com/google/gin-config) to configure experiments. If you wish to run experiments not configured, we strongly advise you to read the [gin user guide](https://github.com/google/gin-config/blob/master/docs/index.md) and use the configs provided as a template.

Other important files:

* `ige/__main__.py`: single entry-point with command line interface. A `--command` specifies a function in `runners.py` to run, while `gin_file` and `gin_param` are used to specify parameters via `gin-config`.
* `ige/runners.py`: contain functions to be run by `__main__.py`. These include training, evaluation and visualizing etc.

If running through `__main__.py` (advised), note all `.gin` files are relative to:

1. `--config_dir` passed as a command line argument; or if not provided,
2. `IGE_CONFIG_DIR` environment variable; or if not defined,
3. `IGE_DIR/config`, where `IGE_DIR` is the location of this repository.

Models are saved to `~/ige_models` by default, but this can be overriden
```
--gin_param='@default_model_dir.base_dir = "~/my/models_dir/"'
```

## Setup

### Installation

Clone this repository and install dependencies

```bash
cd /path/to/parent_dir
git clone https://github.com/jackd/ige.git
cd ige
pip install -r requirements.txt
pip install -e .
```

### Data

We use [tensorflow_datasets](https://github.com/tensorflow/tensorflow_datasets) (`tfds`) for downloading, preprocessing and storing relevant data on disk. The first time you run experiments with a new source of data it may take some time. Pease be patient. In particular:

* downloading `.tar` files creates progress bars that appear frozen, though complete after a time;
* (when it's available) preprocessing high resolution voxel grids and rending images takes a LONG time - particularly for multiple categories. No attempt has been made to multi-thread this operation, but if you need multiple categories you may be best off running multiple instances - one for each category.

Human pose data must be downloaded manually (see below), but everything else should be automated.

#### Human Pose Estimation (HPE)

The data for the human pose estimation is from the [human 3.6m dataset](http://vision.imar.ro/human3.6m/description.php). We use a version made available [here](https://github.com/una-dinosauria/3d-pose-baseline).

Unfortunately, `tfds` does not seem to like automatically downloading this particular data (hosted in dropbox). Until resolved, download the data manually.

For label data and stacked hourglass predictions, use the following.

```bash
H3M_LIFT=~/tensorflow_datasets/downloads/manual/h3m_lift
mkdir -p $H3M_LIFT
wget -O https://www.dropbox.com/s/e35qv3n6zlkouki/h36m.zip $H3M_LIFT
```

Finetuned stacked hourglass predictions are available [here](https://drive.google.com/open?id=0BxWzojlLp259S2FuUXJ6aUNxZkE) to `$H3M_LIFT`. Some users experience issues getting the data with chrome, in which case try using firefox.

If you make use of the data, please cite the works accordingly.

```bibtex
@article{h36m_pami,
  author = {Ionescu, Catalin and Papava, Dragos and Olaru, Vlad and Sminchisescu,    Cristian},
  title = {Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human Sensing   in Natural Environments},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  publisher = {IEEE Computer Society},
  volume = {36},
  number = {7},
  pages = {1325-1339},
  month = {jul},
  year = {2014}
}

@inproceedings{IonescuSminchisescu11,
  author = {Catalin Ionescu, Fuxin Li, Cristian Sminchisescu},
  title = {Latent Structured Models for Human Pose Estimation},
  booktitle = {International Conference on Computer Vision},
  year = {2011}
}

@inproceedings{martinez_2017_3dbaseline,
  title={A simple yet effective baseline for 3d human pose estimation},
  author={Martinez, Julieta and Hossain, Rayat and Romero, Javier and Little, James J.},
  booktitle={ICCV},
  year={2017}
}
```

#### Single View 3D Reconstruction

Preprocessing for [shapenet](https://www.shapenet.org/) dataset/IGE implementation coming soon!

## Tensorflow 2.0

I have aimed to write everything in a manner compatible with `2.0`, but will not be testing/prioritizing support until a stable release.

## Disclaimer

The paper used code based on the `tf.estimator` API. After submission, the tensorflow team announced they would be pushing `tf.keras` rather than `tf.estimator`, so the code was re-written accordingly. As with any re-write, this may have introduced subtle differences. If you believe there is an undocumented difference between the paper and the code, please open an issue. I am yet to re-run all experiments from the paper, so this is entirely possible.

If you would like the original code I am happy to provide it via email. Be warned: it isn't particularly pretty and I won't be supporting issues moving forward.
