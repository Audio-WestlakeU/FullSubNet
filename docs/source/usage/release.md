# Checkpoints and RIRs

## Checkpoints

The [release page](https://github.com/haoxiangsnr/FullSubNet/releases) has two model checkpoints. All checkpoints include "model_state_dict", "optimizer_state_dict", and some other meta information.

The first model checkpoint is the original model checkpoint at the 58th epoch. The performance is shown in this table:

|            | With Reverb   |     |        |       | No Reverb |         |        |       |
|:----------:|:-----------:|:-------:|:------:|:-----:|:---------:|:-------:|:------:|:-----:|
|   Method   |   WB-PESQ   | NB-PESQ | SI-SDR |  STOI |  WB-PESQ  | NB-PESQ | SI-SDR |  STOI |
| FullSubNet |    2.987   |  3.496  | 15.756 | 0.926 |   2.889   |  3.385  | 17.635 | 0.964 |

In addition, some people are interested in the performance when using cumulative normalization. The below one is a pre-trained FullSubNet using cumulative normalization:

|            | With Reverb   |     |        |       | No Reverb |         |        |       |
|:----------:|:-----------:|:-------:|:------:|:-----:|:---------:|:-------:|:------:|:-----:|
|   Method   |   WB-PESQ   | NB-PESQ | SI-SDR |  STOI |  WB-PESQ  | NB-PESQ | SI-SDR |  STOI |
|FullSubNet (Cumulative Norm)|   2.978| 3.503 | 15.820 | 0.928 | 2.863| 3.376 | 17.913 | 0.964 |

If you want to inference or fine-tune based on these checkpoints, please check the usage in the documents.

## Room Impulse Responses

As mentioned in the paper, the room impulse responses (RIRs) come from the Multichannel Impulse Response Database and the Reverb Challenge dataset. Please download the zip package "RIR (Multichannel Impulse Response Database + The REVERB challenge).zip" from the [release page](https://github.com/haoxiangsnr/FullSubNet/releases) if you would like to retrain the FullSubNet.

Note that the zip package includes a folder "rir" and a file "rir.txt." The folder "rir" contains all separated single-channel RIRs extracted from the above two datasets. The suffix (e.g., "m_<n>") of the filename is the index of a microphone. The file "rir.txt" is just a path list of all RIRs. Please modify it to fit your case before you use it.

For some cases, if you would like to extract channel by yourself, you can download these RIRs from pages:
1. [Multichannel Impulse Response Database](https://www.eng.biu.ac.il/~gannot/RIR_DATABASE/)
2. The REVERB challenge data
   1. https://reverb2014.dereverberation.com/tools/reverb_tools_for_Generate_mcTrainData.tgz
   2. https://reverb2014.dereverberation.com/tools/reverb_tools_for_Generate_SimData.tgz