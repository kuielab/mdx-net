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
  * Our spectrogram-based models were trained in two phases (see "How to reproduce the training" for details).
  * For phase 1 we used the MusDB default  86/14 train and validation splits.
  * For phase 2 we used
    * all 100 MusDB trainset tracks for training
    * the MDX challenge public test data for validation
  * Augmentation
    * Random chunking and mixing sources from different tracks ([1])
    * Pitch shift and time stretch ([2])
* Model
  * Blend[1] of two models: TFC-TDF[3] and Demucs[4] 
  * TFC-TDF 
    * Models were trained separately for each source.
    * The input [frequency, time] dimensions are fixed to [2048, 256] for all sources 
      * 256 frames = 6 seconds of audio (sample_rate=44100, hop_length=1024)
      * High frequencies were cut off from the mixture before being input to the networks, and the number of frequency bins to be discarded differs for each source (ex. drums have more high frequencies compared to bass, so cut off more when doing bass separation). 
      * STFT window size differs for each source to fit the frequency dimension of 2048 after cutting off different number of high frequency bins
    * We made the following modifications to the original TFC-TDF model:
      * No densely connected convolutional blocks
      * Multiplicative skip connections
      * Increased depth and  number of hidden channels
    * After training the source-dedicated models we trained an additional network (which we call the 'Mixer') on top of the model outputs, which takes all four estimated sources as input and outputs better estimated sources
      * We only tried a single 1x1 convolution layer for the Mixer, but still gained at least 0.1 SDR for every source on the MDX test set
      * trained without fine-tuning the source-dedicated models
  * Demucs
    * we used the pretrained model with 64 initial hidden channels (not demucs48_hq)
    * overlap=0.5 and no shift trick
  * blending parameters (TFC-TDF : Demucs) => bass 5:5, drums 5:5, other 7:3, vocals 9:1

[1] Stöter, Fabian-Robert, et al. "Open-unmix-a reference implementation for
    music source separation." Journal of Open Source Software 4.41 (2019): 1667.

[2] Cohen-Hadria, Alice, Axel Roebel, and Geoffroy Peeters. "Improving singing voice separation using Deep U-Net and Wave-U-Net with data augmentation." 2019 27th European Signal Processing Conference (EUSIPCO). IEEE, 2019.

[3] Choi, Woosung, et al. "Investigating u-nets with various intermediate blocks for spectrogram-based singing voice separation." 21th International Society for Music Information Retrieval Conference, ISMIR. 2020.

[4] Défossez, Alexandre, et al. "Music source separation in the waveform domain." arXiv preprint arXiv:1911.13254 (2019).


# Reproduction

## How to reproduce the submission

[Leaderboard A](https://gitlab.aicrowd.com/kim_min_seok/demix/tree/submission133)

[Leaderboard B](https://gitlab.aicrowd.com/kim_min_seok/demix/tree/submission106)


## How to reproduce the training

### 1. Data Augmentation

run ```src/utils/data_augmentation.py```

### 2. Phase 1.

- Train ```src.models.mdxnet.ConvTDFNet``` for each source.
  - vocals: ```python run.py experiment=multigpu_vocals model=ConvTDFNet_vocals```
  - drums: ```python run.py experiment=multigpu_drums model=ConvTDFNet_drums```
  - bass: ```python run.py experiment=multigpu_bass```
  - other: ```python run.py experiment=multigpu_other model=ConvTDFNet_other```

- for training, each takes at least 3 days.

### Phase 2

- Train Mixer
  - locate candidate checkpoints by appending ```ckpt``` variable in the ```yaml``` config file.
  - train ```from src.models.mdxnet Mixer ```

# License

[MIT Licence](LICENSE.MD)