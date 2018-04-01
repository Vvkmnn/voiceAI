# voiceAI

The third Capstone project as part of the [Artificial Intelligence Nanodegree](https://www.udacity.com/course/artificial-intelligence-nanodegree--nd889), and focusing on **Spectrograms**, **Voice User Interfaces**, **Recurrent Neural Nets**, and **Speech Recognition**!

## Overview

In this notebook, we will build a deep neural network that functions as part of an end-to-end automatic speech recognition (ASR) pipeline (!): 

![](./images/pipeline.png)

- **STEP 1: PRE-PROCESSING:** Converts raw audio to one of two feature representations that are commonly used for ASR. 
- **STEP 2: ACOUSTIC MODEL:** Accept transformed audio features as input and return a probability distribution over all potential transcriptions, picking it's best guess (using a variety of models!)
- **STEP 3: PREDICTION:** Lastly, the pipeline takes the output from the acoustic model and returns a predicted transcription for validation.

## Models

We will use the [LibriSpeech dataset](http://www.openslr.org/12/) to train and evaluate our models, namely: 

#### Model 0: RNN

#### Model 1: RNN + TimeDistributed Dense

![](./images/rnn_model.png)

![](./images/rnn_model_unrolled.png)

#### Model 2: CNN + RNN + TimeDistributed Dense

![](./images/cnn_rnn_model.png)

#### Model 3: Deeper RNN + TimeDistributed Dense

![](./images/deep_rnn_model.png)

#### Model 4: Bidirectional RNN + TimeDistributed Dense

![](./images/bidirectional_rnn_model.png)

#### Model 5: Deep Bidirectional RNN + TimeDistributed             

#### Model 6: Deep Bidirectional RNN + TimeDistributed with Dropout

#### Model 8: CNN + RNN + TimeDistributed with Dropout              

## Setup

Run via [Amazon Elastic Compute Cloud](https://aws.amazon.com/ec2/), using [The Deep Learning AMI with Cuda Support!](https://aws.amazon.com/marketplace/fulfillment?productId=8011986f-8b40-4ce3-9eed-1f877ce4d941&ref_=dtl_psb_continue) on a `p2.xlarge` GPU instance: 

First, prepping the instance with [Tensorflow]((https://www.tensorflow.org/) and friends, and the audio processing library; [`libav`](https://libav.org/download/):
```
sudo python3 -m pip install tensorflow-gpu==1.1 udacity-pa tqdm
sudo apt-get install libav-tools
sudo python3 -m pip install python_speech_features librosa soundfile
install libav
```

Obtain the appropriate subsets of the LibriSpeech dataset, and convert all flac files to wav format.
```
wget http://www.openslr.org/resources/12/dev-clean.tar.gz
tar -xzvf dev-clean.tar.gz
wget http://www.openslr.org/resources/12/test-clean.tar.gz
tar -xzvf test-clean.tar.gz
mv flac_to_wav.sh LibriSpeech
cd LibriSpeech
./flac_to_wav.sh
```

Create JSON files corresponding to the train and validation datasets.
```
cd ..
python create_desc_json.py LibriSpeech/dev-clean/ train_corpus.json
python create_desc_json.py LibriSpeech/test-clean/ valid_corpus.json
```

(*Optional*) Setup local environment
```
conda create --name voiceAI
source activate voiceai
pip install -r requirements.txt
pip install tensorflow-gpu==1.1.0
```

Start Jupyter, and connect via your IPv4 address:
```
jupyter notebook --ip=0.0.0.0 --no-browser
```

## Thanks

[Udacity](@udacity) borrowed the `create_desc_json.py` and `flac_to_wav.sh` files from the [ba-dls-deepspeech](https://github.com/baidu-research/ba-dls-deepspeech) repository, along with some functions used to generate spectrograms.
