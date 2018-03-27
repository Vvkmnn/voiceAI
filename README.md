# voiceAI

The final capstone in the AIND, dealing with **Voice User Interfaces** and **Speech Recognition via Neural Nets!**.

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

#### Model 2: CNN + RNN + TimeDistributed Dense

#### Model 3: Deeper RNN + TimeDistributed Dense

#### Model 4: Bidirectional RNN + TimeDistributed Dense

#### Model 5: Final Model


**TODO**: Fix this as project is completed. 
![](./images/rnn_model.png)
![](./images/rnn_model_unrolled.png)
![](./images/bidirectional_rnn_model.png)
![](./images/cnn_rnn_model.png)
![](./images/deep_rnn_model.png)
![](./images/simple_rnn.png)
![](./images/simple_rnn_unrolled.png)

## Setup

Install the [`libav` package](https://libav.org/download/):
```
brew install libav
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

Setup local environment
```
conda create --name voiceAI
source activate voiceai
pip install -r requirements.txt
pip install tensorflow-gpu==1.1.0
```

Start Jupyter (and make sure you're using your environment kernel, not your system kernel)
```
python -m ipykernel install --user --name myenv --display-name "Python (voiceAI)"
python -c "from keras import backend"
jupyter notebook 
```

![select aind-vui kernel](./images/select_kernel.png)

## Thanks

[Udacity](@udacity) borrowed the `create_desc_json.py` and `flac_to_wav.sh` files from the [ba-dls-deepspeech](https://github.com/baidu-research/ba-dls-deepspeech) repository, along with some functions used to generate spectrograms.
