# Unsupervised Sentence Summarization

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Unsupervised sentence summarization by contextual matching.

This is the code for the paper: \
[Simple Unsupervised Summarization by Contextual Matching](https://arxiv.org/pdf/1907.13337.pdf) (ACL 2019) \
Jiawei Zhou, Alexander Rush

<img src=figure1.png>
<!---<img src=figure1.png width="500" height="500">--->


## Overview

Using contextual word embeddings (e.g. pre-trained ELMo model) along with a language model trained on summary style sentences, we are able to find sentence level summarizations in an unsupervised way without being exposed to any paired data.

The summary generation process is through beam search to maximize a product-of-expert score which is a combination of a contextual matching model (relying on pre-trained left-contextual embeddings) and a language fluency model (based on the summary domain specific language model). This works for both abstractive and extractive sentence level summarization.

**Note:** as our generation process is left-to-right, we used only the forward model in ELMo. We also tested on the small version of GPT-2 after its release, but found it didn't not perform as well as ELMo in our setup.


## Dependencies

The code was based and tested on the following libraries:
- python 3.6
- PyTorch 0.4.1
- allennlp 0.5.1

For Rouge evaluation, we used [files2rouge](https://github.com/pltrdy/files2rouge).


## Datasets & Summary Results & Pre-trained Language Models

| Data & Task  | Test Set & <br>Unsupervised Model Output | Summary LM & Vocabulary | Full Dataset |
|:---:|:---:|:---:|:---:|
| English Gigaword <br>(abstractive summarization) | [test data](./data/gigaword) <br> [model output](./results_elmo_giga) | [language model](https://drive.google.com/file/d/1iF0tLvoo74-o22-1jUjMTrLwK948sMKp/view?usp=sharing) | [full data](https://github.com/harvardnlp/sent-summary) |
| Google Sentence Compression <br>(extractive summarization) | [test data](./data/sentence_compression) <br> [model output](./results_elmo_sc)| [language model](https://drive.google.com/file/d/1KVh7J6Mpj6W5YFV0DPAb81OwJSo26C7g/view?usp=sharing) | [full data](https://github.com/google-research-datasets/sentence-compression) |

## Unsupervised Summary Generation

To generate summaries for a given corpus of source sentences, make sure the following two components are prepared:
- The ELMo model contained in the [allennlp](https://github.com/allenai/allennlp) library package
- A pre-trained LSTM based language model on the summary style short sentences (we have included our language modeling and training scripts in [lm_lstm](./lm_lstm), as well as our pre-trained models [above](#Datasets-&-Summary-Results-&-Pre-trained-Language-Models))

---

Suppose the file structure is as follows:
```
├── ./
├── data/
    ├── gigaword/
    ├── sentence_compression/
├── lm_lstm/
├── lm_lstm_models/
    ├── gigaword/
    ├── sentence_compression/
├── uss/
    ├── ...
├── ...
```

where we use two datasets as examples, English Gigaword for abstractive sentence summarization and Google sentence compression dataset for extractive sentence summarization, as were used in the paper. Suppose these data are stored in the `./data/` directory.

For the following commands we take the [English Gigaword dataset](https://github.com/harvardnlp/sent-summary) as an example.

---

**To train a summary domain specific language model:**

```
python lm_lstm/main.py --data_src user --userdata_path ./data/gigaword --userdata_train train.title.txt --userdata_val valid.title.filter.txt --userdata_test task1_ref0_unk.txt --bptt 32 --bsz 256 --embedsz 1024 --hiddensz 1024 --tieweights 0 --optim SGD --lr 0.1 --gradclip 15 --epochs 50 --vocabsave ./lm_lstm_models/gigaword/vocabTle.pkl --save ./lm_lstm_models/gigaword/Tle_LSTM_untied.pth --devid 0
```
Remember to change and check the data file paths and names, and save the vocabulary and model to proper places. For a full list of hyperparameters and their meanings use `python lm_lstm/main.py --help` or check the python script.

**A minor detail**: in the processed [English Gigaword dataset](https://github.com/harvardnlp/sent-summary) we used, the training and validation sets contain unknown words as `<unk>`, whereas in the test set they are represented as `UNK`. When training the language model we replace the `UNK` tokens in the test summary set by `<unk>`. And after generating the test summaries, to compare with the original reference, we map `<unk>` back to `UNK` for Rouge evaluation to be consistent with the literature.

In our experiments, the above command should produce a language model which achieves train perplexity ~62, validation perplexity ~72, and test perplexity ~201 (as the test set if quite different from the train and validation sets). And training takes about a day on a DGX V100 GPU.

---

After obtaining the summary domain specific language model, we can do the **summary generation using the following command:**

```
python uss/summary_search_elmo.py --src ./data/gigaword/input_unk.txt --modelclass ./lm_lstm --model ./lm_lstm_models/gigaword/Tle_LSTM_untied.pth --vocab ./lm_lstm_models/gigaword/vocabTle.pkl --n 6 --ns 10 --nf 300 --elmo_layer cat --alpha 0.1 --beta 0 --beam_width 10 --devid 0 --save_dir ./results_elmo_giga/
```

where:
- `--src` specifies the sentence corpus to be summarized
- `--modelclass` is the directory in which the language model source script is saved
- `--model` is the path of the language model to be used
- `--vocab` is the vocabulary file associated with the language model

and finally the results will be saved in the directory `--save_dir` with a system generated file name based on user specified hyperparameters. For the full list of hyperparameters for the summary generation process and their meanings (alghough most of them were used for experimental purposes and need not to be changed), use `python uss/summary_search_elmo.py --help` or check the python script.

With the above command exactly, a file named "smry_input_unk_Ks10_clust1_ELcat_eosavg0_n6_ns10_nf300_a0.1_b0.0_all.txt" containing all the generated sentence summarizations will be saved in the directory "./results_elmo_giga/". In this file, for each source sentence, all of the finished hypotheses from beam search are saved as candidate summary sentences, along with their alignments to the original source sentence, as well as the combined scores, contextual matching scores, and language modeling scores. Note that this searching process is relatively slow, as we need to calculate the contextual embeddings for every sentence prefix and every candidate next word in the procedure, even with our optimization of caching and batching.


## Evaluation

To be consistent with the literature for evaluation, we need to select one summary sentence from the list to compare with the reference summary and calculate some metric statistics such as Rouge scores. Since our generation is unsupervised it could be difficult to select the best summary from a list of candidate summaries, and it is often the case that there is a better one than our selected one. Nevertheless we use a simple length penalized beam search score for our selection criterion.

**For summary selection and evaluation, run the following command:**

```
python uss/summary_select_eval.py --src ./data/gigaword/input_unk.txt --ref ./data/gigaword/task1_ref0.txt --gen ./results_elmo_giga/smry_input_unk_Ks10_clust1_ELcat_eosavg0_n6_ns10_nf300_a0.1_b0.0_all.txt --save_dir ./results_elmo_giga --lp 0.1
```

where:
- `--src`: source sentences file path
- `--ref`: reference sentences file path
- `--gen`: generated summary sentences file path
- `--save_dir`: directory to save selected summaries
- `--lp`: additive length penalty (usually between -0.1 and 0.1)

This will generate a file named "smry_input_unk_Ks10_clust1_ELcat_eosavg0_n6_ns10_nf300_a0.1_b1.0_single.txt" in "./results_elmo_giga/", containing a single summary selected for each of the source sentences. And Rouge scores will be computed and printed, along with other statistics including copy rate, compression rate, and average summary length.

Note that the Rouge evaluation is based on [files2rouge](https://github.com/pltrdy/files2rouge).

## Data and Sample Output

We have included the test sets of English Gigaword dataset and Google sentence compression evaluation set in the "./data" folder.

We also include the summary outputs from our unsupervised method for these two test sets in "./results_elmo_giga" and "./results_elmo_sc" respectively.

## Citing

If you find the resources in this repository useful, please consider citing:

```
@inproceedings{zhou2019simple,
  title={Simple Unsupervised Summarization by Contextual Matching},
  author={Zhou, Jiawei and Rush, Alexander M},
  booktitle={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  pages={5101--5106},
  year={2019}
}
```
