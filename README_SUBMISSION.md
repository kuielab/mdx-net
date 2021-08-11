# Submission

## Submission Summary

* Submission ID: 151249
* Submitter: kim_min_seok
* Final rank: 3rd place on leaderboard B
* Final scores on MDXDB21:


  | SDR_song | SDR_bass | SDR_drums | SDR_other | SDR_vocals |
  | :------: | :------: | :-------: | :-------: | :--------: |
  |   7.37   |   7.50   |   7.55    |   5.53    |    8.90    |


## Model Summary

* Data
  * We used the MusDB default 14 track validation set for validation, and the other 136 tracks for training.
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
    > note: our final submission for leaderboard B was submitted before training converged, and the learning rate might have not been optimal as well. Furthermore, the bass separation model doesn't even use external data.
  * Demucs
    * we used the pretrained model with extra data (demucs_extra)
    * overlap=0.5 and no shift trick
  * blending parameters (TFC-TDF : Demucs) => bass 5:5, drums 5:5, other 7:3, vocals 9:1

[1] S. Uhlich et al., "Improving music source separation based on deep neural networks through data augmentation and network blending," 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2017.

[2] Cohen-Hadria, Alice, Axel Roebel, and Geoffroy Peeters. "Improving singing voice separation using Deep U-Net and Wave-U-Net with data augmentation." 2019 27th European Signal Processing Conference (EUSIPCO). IEEE, 2019.

[3] Choi, Woosung, et al. "Investigating u-nets with various intermediate blocks for spectrogram-based singing voice separation." 21th International Society for Music Information Retrieval Conference, ISMIR. 2020.

[4] Défossez, Alexandre, et al. "Music source separation in the waveform domain." arXiv preprint arXiv:1911.13254 (2019).


# Reproduction

## How to reproduce the submission

***Note***: The inference time is very close to the time limit, so submission will randomly fail. You might have to submit it several times.

- obtain ```.onnx``` files as described in the [following section](#how-to-reproduce-the-training)
- follow this instruction to deploy parameters
    ```
    git clone https://github.com/kuielab/mdx-net-submission.git
    cd mdx-net-submission
    git checkout leaderboard_B
    git lfs install
    mv ${*.onnx} onnx/
    ```
- or visit the following links that hold the pretrained ```.onnx``` files.
  - [Leaderboard B](https://github.com/kuielab/mdx-net-submission/tree/leaderboard_B)

- or visit the submitted repository
  - [Leaderboard B](https://gitlab.aicrowd.com/kim_min_seok/demix/tree/submission106)


## How to reproduce the training

### 1. Data Preparation

Pitch Shift and Time Stretch [2]
- This could have been done on-the-fly along with chunking and mixing ([1]), but we preferred faster train steps over less disk usage. The following scripts are for saving augmented tracks to disk before training. 

- For Leaderboard B
    - run ```python src/utils/data_augmentation.py --data_dir ${your_musdb_path} --train True --test True``` 
    - (takes 3~4 hours)

### 2. Phase 1

- Train ```src.models.mdxnet.ConvTDFNet``` for each source.
  - vocals: ```python run.py experiment=multigpu_vocals model=ConvTDFNet_vocals```
  - drums: ```python run.py experiment=multigpu_drums model=ConvTDFNet_drums```
  - bass: ```python run.py experiment=multigpu_bass model=ConvTDFNet_bass```
  - other: ```python run.py experiment=multigpu_other model=ConvTDFNet_other``` 
  
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

- After training the 4 models, rename the best ```.onnx``` files to '{source_name}.onnx' for each source ('vocals.onnx', 'bass.onnx', etc.), then copy them to the 'onnx' directory in the [submission repository](https://github.com/kuielab/mdx-net-submission/tree/leaderboard_B/model)

### 3. Phase 2

We did not train ```Mixer``` for the model used in Leaderboard B.

# License

[MIT Licence](LICENSE.MD)
