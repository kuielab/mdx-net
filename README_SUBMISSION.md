# Submission

## Submission Summary

* Submission ID: 151907
* Submitter: kim_min_seok
* Final rank: 2nd place on leaderboard A
* Final scores on MDXDB21:

  | SDR_song | SDR_bass | SDR_drums | SDR_other | SDR_vocals |
  | :------: | :------: | :-------: | :-------: | :--------: |
  |   7.24   |   7.23   |   7.17    |   5.64    |    8.90    |

## Model Summary

* Data
  * We used the MusDB default 86/14 train and validation splits.
  * Augmentation
    * Random chunking and mixing sources from different tracks ([1])
    * Pitch shift and time stretch ([2])
* Model
  * Blend[1] of two models: a modified version of TFC-TDF[3] and Demucs[4] 
  * TFC-TDF 
    * Models were trained separately for each source.
    * The input [frequency, time] dimensions are fixed to [2048, 256] for all sources 
      * 256 frames = 6 seconds of audio (sample_rate=44100, hop_length=1024)
      * High frequencies were cut off from the mixture before being input to the networks, and the number of frequency bins to be discarded differs for each source (ex. drums have more high frequencies compared to bass, so cut off more when doing bass separation). In order to fit the frequency dimension of 2048, n_fft differs for each source.
    * We made the following modifications to the original TFC-TDF model:
      * No densely connected convolutional blocks
      * Multiplicative skip connections
      * Increased depth and number of hidden channels
    * After training the per-source models we trained an additional network (which we call the 'Mixer') on top of the model outputs, which takes all four estimated sources as input and outputs better estimated sources
      * We only tried a single 1x1 convolution layer for the Mixer (due to inference time limit), but still gained at least 0.1 SDR for every source on MDXDB21.
  * Demucs
    * we used the pretrained model with 64 initial hidden channels (not demucs48_hq)
    * overlap=0.5 and no shift trick
  * blending parameters (TFC-TDF : Demucs) => bass 5:5, drums 5:5, other 7:3, vocals 9:1

[1] S. Uhlich et al., "Improving music source separation based on deep neural networks through data augmentation and network blending," 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2017.

[2] Cohen-Hadria, Alice, Axel Roebel, and Geoffroy Peeters. "Improving singing voice separation using Deep U-Net and Wave-U-Net with data augmentation." 2019 27th European Signal Processing Conference (EUSIPCO). IEEE, 2019.

[3] Choi, Woosung, et al. "Investigating u-nets with various intermediate blocks for spectrogram-based singing voice separation." 21th International Society for Music Information Retrieval Conference, ISMIR. 2020.

[4] Défossez, Alexandre, et al. "Music source separation in the waveform domain." arXiv preprint arXiv:1911.13254 (2019).


# Reproduction

## How to reproduce the submission

***Note***: The inference time is very close to the time limit, so submission will randomly fail. You might have to submit it several times.

- obtain ```.onnx``` files and ```.ckpt``` file as described in the [following section](#how-to-reproduce-the-training)
- follow this instruction to deploy parameters
    ```
    git clone https://github.com/kuielab/mdx-net-submission.git
    cd mdx-net-submission
    git checkout leaderboard_A
    git lfs install
    mv ${*.onnx} onnx/
    mv ${*.ckpt} model/  
    ```
- or visit the following links that hold the pretrained ```.onnx``` files and ```.ckpt``` file
  - [Leaderboard A](https://github.com/kuielab/mdx-net-submission/tree/leaderboard_A)

- or visit the submitted repository
  - [Leaderboard A](https://gitlab.aicrowd.com/kim_min_seok/demix/tree/submission133)


## How to reproduce the training

### 1. Data Preparation

Pitch Shift and Time Stretch [2]
- This could have been done on-the-fly along with chunking and mixing ([1]), but we preferred faster train steps over less disk usage. The following scripts are for saving augmented tracks to disk before training. 

- For Leaderboard A
    - run ```python src/utils/data_augmentation.py --data_dir ${your_musdb_path} --train True --test False```

### 2. Phase 1

- Train ```src.models.mdxnet.ConvTDFNet``` for each source.
  - vocals: ```python run.py experiment=multigpu_vocals model=ConvTDFNet_vocals```
  - drums: ```python run.py experiment=multigpu_drums model=ConvTDFNet_drums```
  - bass: ```python run.py experiment=multigpu_bass model=ConvTDFNet_bass```
  - other: ```python run.py experiment=multigpu_other model=ConvTDFNet_other```

- For training, each takes at least 3 days, usually 4~5 days to early-stop for the current configurations. 
  
- Default logging system is [wandb](https://www.wandb.com/)
  ![](val_loss_vocals.png)  
  
- Checkpoint result saving callbacks
  - We use [onnx](https://onnx.ai/) for faster inference to meet the time limit
    - see the [related issue](https://github.com/ws-choi/Conditioned-Source-Separation-LaSAFT/issues/20#issuecomment-840407759)
  - You don't have to manually convert ```.onnx``` files. Our code automatically generates ```.onnx``` whenever a new checkpoint is saved by [checkpoint callback](https://github.com/kuielab/mdx-net/blob/7c6f7daecde13c0e8ed97f308577f6690b0c31af/configs/callbacks/default.yaml#L2)  
    ![](onnx_callback.png)
  - This function was implemented as a callback function
    - see [this](https://github.com/kuielab/mdx-net/blob/7c6f7daecde13c0e8ed97f308577f6690b0c31af/configs/callbacks/default.yaml#L18)
    - and [this](https://github.com/kuielab/mdx-net/blob/7c6f7daecde13c0e8ed97f308577f6690b0c31af/src/callbacks/onnx_callback.py#L11)

- After training the 4 models, rename the best ```.onnx``` files to '{source_name}.onnx' for each source ('vocals.onnx', 'bass.onnx', etc.), then copy them to the 'onnx' directory in the [submission repository](https://github.com/kuielab/mdx-net-submission/tree/leaderboard_A/model)

#### The epoch of each checkpoint we used  
  - vocals: 2360 epoch
  - bass: 1720 epoch
  - drums: 600 epoch
  - other: 1720 epoch

### 3. Phase 2

This phase **does not fine-tune** the pretrained separators from the previous phase.

- Train Mixer
  - fill in the pretrained ckpt path for each source in [Mixer.yaml](https://github.com/kuielab/mdx-net/blob/8cabde1cb803b0696ec88570a2e8d113b72d9c55/configs/model/Mixer.yaml#L12)
  - run ```python run.py experiment=mixer model=Mixer```
  - Mixer is a very small model
      - the number of learnable parameters in ```Mixer``` < 100
      - so we do not use ```onnx``` in this case
      - , and no need to wait too much for the convergence (~ 10 epoch).
  - pick the top Mixer ckpt and rename it to 'mixer.pt', then copy it to the 'model' directory in the [submission repository](https://github.com/kuielab/mdx-net-submission/tree/leaderboard_A/model)
  
  
# License

[MIT Licence](LICENSE.MD)
