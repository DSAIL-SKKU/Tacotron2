# Tacotron2

Korean Speech Synthesis with Tacotron

Note that this repo is based on https://github.com/hccho2/Tacotron2-Wavenet-Korean-TTS, https://github.com/keithito/tacotron


## Background

In February 2018, Google published a paper, [NATURAL TTS SYNTHESIS BY CONDITIONINGWAVENET ON MEL SPECTROGRAM PREDICTIONS],
where they present a neural text-to-speech model that learns to synthesize speech directly from
(text, audio) pairs. However, they didn't release their source code or training data. This is an
independent attempt to provide an open-source implementation of the model described in their paper.

The quality isn't as good as Google's demo yet, but hopefully it will get there someday :-).
Pull requests are welcome!



## Quick Start

### Installing dependencies

1. Install Python 3.

2. Install the latest version of [TensorFlow](https://www.tensorflow.org/install/) for your platform. For better
   performance, install with GPU support if it's available. This code works with TensorFlow 1.3 and later.

3. Install requirements:
   ```
   pip install -r requirements.txt
   ```


### Training

*Note: you need at least 40GB of free disk space to train a model.*

1. **Download a speech dataset.**

   The following are supported out of the box:
    * [KSS Dataset](https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset) (Public Domain)

   You can use other datasets if you convert them to the right format. See [TRAINING_DATA.md](TRAINING_DATA.md) for more info.


2. **Unpack the dataset into `~/tacotron`**

   After unpacking, your tree should look like this for LJ Speech:
   ```
   tacotron
     |- kss
         |- metadata.csv
         |- wavs
   ```


3. **Preprocess the data**
   ```
   python3 preprocess.py --dataset kss
   ```

4. **Train a model**
   ```
   python3 train.py
   ```

   Tunable hyperparameters are found in [hparams.py](hparams.py). You can adjust these at the command
   line using the `--hparams` flag, for example `--hparams="batch_size=16,outputs_per_step=2"`.
   Hyperparameters should generally be set to the same values at both training and eval time.
   The default hyperparameters are recommended for LJ Speech and other English-language data.
   See [TRAINING_DATA.md](TRAINING_DATA.md) for other languages.


5. **Monitor with Tensorboard** (optional)
   ```
   tensorboard --logdir ~/tacotron/logs-tacotron
   ```

   The trainer dumps audio and alignments every 1000 steps. You can find these in
   `~/tacotron/logs-tacotron`.

6. **Synthesize from a checkpoint**
   ```
   python3 demo_server.py --checkpoint ~/tacotron/logs-tacotron/model.ckpt-185000
   ```
   Replace "185000" with the checkpoint number that you want to use, then open a browser
   to `localhost:9000` and type what you want to speak. Alternately, you can
   run [eval.py](eval.py) at the command line:
   ```
   python3 eval.py --checkpoint ~/tacotron/logs-tacotron/model.ckpt-185000
   ```
   If you set the `--hparams` flag when training, set the same value here.


## Modifications

  * We add Stepwise Monotonic Attention, Monotonic Attention, GMM Attention, Loung Attention (20.01.20)
 
