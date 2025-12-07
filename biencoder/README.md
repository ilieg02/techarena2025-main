---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:810276
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-mpnet-base-v2
widget:
- source_sentence: 仔,你饿吗
  sentences:
  - 我是capf出生
  - 你能给我更多关于自动化分析工具的建议吗？
  - 仔,你饿吗
- source_sentence: 如何证明代数基本定理
  sentences:
  - 爷爷
  - 如何证明代数基本定理
  - 他支教去了
- source_sentence: 我想买东西
  sentences:
  - 我需要一个可以帮助我快速制作PPT的工具，有什么推荐吗？
  - 炒了
  - 我想买东西
- source_sentence: MOSS是否可以帮助企业进行竞争对手分析？
  sentences:
  - MOSS是否可以帮助企业进行竞争对手分析？
  - 太贱了你
  - 明恋多尴尬
- source_sentence: 我应该如何制定一个合理的预算？
  sentences:
  - 我应该如何制定一个合理的预算？
  - 有木有减肥法
  - 如何在团队中更好地协作和沟通？
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-mpnet-base-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) <!-- at revision e8c3b32edf5434bc2275fc9bab85f82640a19130 -->
- **Maximum Sequence Length:** 384 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 384, 'do_lower_case': False, 'architecture': 'MPNetModel'})
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    '我应该如何制定一个合理的预算？',
    '我应该如何制定一个合理的预算？',
    '如何在团队中更好地协作和沟通？',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 1.0000, 0.0004],
#         [1.0000, 1.0000, 0.0004],
#         [0.0004, 0.0004, 1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 810,276 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                        | sentence_1                                                                        |
  |:--------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                            |
  | details | <ul><li>min: 3 tokens</li><li>mean: 13.27 tokens</li><li>max: 93 tokens</li></ul> | <ul><li>min: 3 tokens</li><li>mean: 13.27 tokens</li><li>max: 93 tokens</li></ul> |
* Samples:
  | sentence_0                              | sentence_1                              |
  |:----------------------------------------|:----------------------------------------|
  | <code>MOSS 有什么建议可以用来帮助我实现目标？</code>     | <code>MOSS 有什么建议可以用来帮助我实现目标？</code>     |
  | <code>你是速生鸡吗</code>                     | <code>你是速生鸡吗</code>                     |
  | <code>我在学习编程时遇到了一个问题，如何能够更快地解决它？</code> | <code>我在学习编程时遇到了一个问题，如何能够更快地解决它？</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 1
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 1
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
<details><summary>Click to expand</summary>

| Epoch  | Step  | Training Loss |
|:------:|:-----:|:-------------:|
| 0.0099 | 500   | 0.061         |
| 0.0197 | 1000  | 0.0518        |
| 0.0296 | 1500  | 0.0487        |
| 0.0395 | 2000  | 0.0475        |
| 0.0494 | 2500  | 0.0445        |
| 0.0592 | 3000  | 0.0458        |
| 0.0691 | 3500  | 0.0479        |
| 0.0790 | 4000  | 0.0494        |
| 0.0889 | 4500  | 0.0418        |
| 0.0987 | 5000  | 0.0392        |
| 0.1086 | 5500  | 0.0436        |
| 0.1185 | 6000  | 0.0399        |
| 0.1283 | 6500  | 0.0428        |
| 0.1382 | 7000  | 0.0476        |
| 0.1481 | 7500  | 0.0456        |
| 0.1580 | 8000  | 0.0422        |
| 0.1678 | 8500  | 0.0493        |
| 0.1777 | 9000  | 0.0451        |
| 0.1876 | 9500  | 0.0423        |
| 0.1975 | 10000 | 0.0438        |
| 0.2073 | 10500 | 0.0464        |
| 0.2172 | 11000 | 0.0463        |
| 0.2271 | 11500 | 0.0449        |
| 0.2370 | 12000 | 0.0417        |
| 0.2468 | 12500 | 0.0418        |
| 0.2567 | 13000 | 0.0364        |
| 0.2666 | 13500 | 0.0464        |
| 0.2764 | 14000 | 0.0412        |
| 0.2863 | 14500 | 0.044         |
| 0.2962 | 15000 | 0.0362        |
| 0.3061 | 15500 | 0.0437        |
| 0.3159 | 16000 | 0.0382        |
| 0.3258 | 16500 | 0.0434        |
| 0.3357 | 17000 | 0.0435        |
| 0.3456 | 17500 | 0.0414        |
| 0.3554 | 18000 | 0.0494        |
| 0.3653 | 18500 | 0.0439        |
| 0.3752 | 19000 | 0.0364        |
| 0.3850 | 19500 | 0.0398        |
| 0.3949 | 20000 | 0.0376        |
| 0.4048 | 20500 | 0.0419        |
| 0.4147 | 21000 | 0.0432        |
| 0.4245 | 21500 | 0.0415        |
| 0.4344 | 22000 | 0.0439        |
| 0.4443 | 22500 | 0.0434        |
| 0.4542 | 23000 | 0.04          |
| 0.4640 | 23500 | 0.0454        |
| 0.4739 | 24000 | 0.0392        |
| 0.4838 | 24500 | 0.0379        |
| 0.4937 | 25000 | 0.048         |
| 0.5035 | 25500 | 0.0428        |
| 0.5134 | 26000 | 0.0378        |
| 0.5233 | 26500 | 0.0424        |
| 0.5331 | 27000 | 0.0414        |
| 0.5430 | 27500 | 0.0474        |
| 0.5529 | 28000 | 0.0397        |
| 0.5628 | 28500 | 0.0425        |
| 0.5726 | 29000 | 0.0423        |
| 0.5825 | 29500 | 0.0445        |
| 0.5924 | 30000 | 0.0414        |
| 0.6023 | 30500 | 0.0453        |
| 0.6121 | 31000 | 0.0406        |
| 0.6220 | 31500 | 0.0441        |
| 0.6319 | 32000 | 0.0452        |
| 0.6417 | 32500 | 0.0407        |
| 0.6516 | 33000 | 0.0424        |
| 0.6615 | 33500 | 0.0396        |
| 0.6714 | 34000 | 0.0465        |
| 0.6812 | 34500 | 0.0403        |
| 0.6911 | 35000 | 0.0425        |
| 0.7010 | 35500 | 0.0501        |
| 0.7109 | 36000 | 0.0444        |
| 0.7207 | 36500 | 0.0403        |
| 0.7306 | 37000 | 0.0395        |
| 0.7405 | 37500 | 0.0428        |
| 0.7504 | 38000 | 0.0393        |
| 0.7602 | 38500 | 0.0409        |
| 0.7701 | 39000 | 0.0406        |
| 0.7800 | 39500 | 0.0409        |
| 0.7898 | 40000 | 0.0434        |
| 0.7997 | 40500 | 0.0458        |
| 0.8096 | 41000 | 0.0451        |
| 0.8195 | 41500 | 0.0441        |
| 0.8293 | 42000 | 0.0361        |
| 0.8392 | 42500 | 0.0409        |
| 0.8491 | 43000 | 0.042         |
| 0.8590 | 43500 | 0.0369        |
| 0.8688 | 44000 | 0.0395        |
| 0.8787 | 44500 | 0.045         |
| 0.8886 | 45000 | 0.0352        |
| 0.8984 | 45500 | 0.0455        |
| 0.9083 | 46000 | 0.0454        |
| 0.9182 | 46500 | 0.042         |
| 0.9281 | 47000 | 0.0425        |
| 0.9379 | 47500 | 0.0377        |
| 0.9478 | 48000 | 0.0426        |
| 0.9577 | 48500 | 0.043         |
| 0.9676 | 49000 | 0.0417        |
| 0.9774 | 49500 | 0.041         |
| 0.9873 | 50000 | 0.0419        |
| 0.9972 | 50500 | 0.0424        |
| 0.0099 | 500   | 0.04          |
| 0.0197 | 1000  | 0.0445        |
| 0.0296 | 1500  | 0.0475        |
| 0.0395 | 2000  | 0.0407        |
| 0.0494 | 2500  | 0.0445        |
| 0.0592 | 3000  | 0.0404        |
| 0.0691 | 3500  | 0.0451        |
| 0.0790 | 4000  | 0.0435        |
| 0.0889 | 4500  | 0.0435        |
| 0.0987 | 5000  | 0.0463        |
| 0.1086 | 5500  | 0.0444        |
| 0.1185 | 6000  | 0.0479        |
| 0.1283 | 6500  | 0.0425        |
| 0.1382 | 7000  | 0.0434        |
| 0.1481 | 7500  | 0.041         |
| 0.1580 | 8000  | 0.0455        |
| 0.1678 | 8500  | 0.0417        |
| 0.1777 | 9000  | 0.0504        |
| 0.1876 | 9500  | 0.0399        |
| 0.1975 | 10000 | 0.0417        |
| 0.2073 | 10500 | 0.0514        |
| 0.2172 | 11000 | 0.0437        |
| 0.2271 | 11500 | 0.0391        |
| 0.2370 | 12000 | 0.0419        |
| 0.2468 | 12500 | 0.0433        |
| 0.2567 | 13000 | 0.0459        |
| 0.2666 | 13500 | 0.0434        |
| 0.2764 | 14000 | 0.0423        |
| 0.2863 | 14500 | 0.0374        |
| 0.2962 | 15000 | 0.0447        |
| 0.3061 | 15500 | 0.0385        |
| 0.3159 | 16000 | 0.0414        |
| 0.3258 | 16500 | 0.0466        |
| 0.3357 | 17000 | 0.0383        |
| 0.3456 | 17500 | 0.0404        |
| 0.3554 | 18000 | 0.0409        |
| 0.3653 | 18500 | 0.0441        |
| 0.3752 | 19000 | 0.0463        |
| 0.3850 | 19500 | 0.0443        |
| 0.3949 | 20000 | 0.0435        |
| 0.4048 | 20500 | 0.0386        |
| 0.4147 | 21000 | 0.0457        |
| 0.4245 | 21500 | 0.0417        |
| 0.4344 | 22000 | 0.0438        |
| 0.4443 | 22500 | 0.041         |
| 0.4542 | 23000 | 0.0415        |
| 0.4640 | 23500 | 0.0441        |
| 0.4739 | 24000 | 0.0437        |
| 0.4838 | 24500 | 0.0433        |
| 0.4937 | 25000 | 0.0468        |
| 0.5035 | 25500 | 0.0438        |
| 0.5134 | 26000 | 0.0462        |
| 0.5233 | 26500 | 0.0428        |
| 0.5331 | 27000 | 0.0484        |
| 0.5430 | 27500 | 0.0417        |
| 0.5529 | 28000 | 0.0382        |
| 0.5628 | 28500 | 0.0367        |
| 0.5726 | 29000 | 0.0416        |
| 0.5825 | 29500 | 0.0464        |
| 0.5924 | 30000 | 0.0431        |
| 0.6023 | 30500 | 0.0336        |
| 0.6121 | 31000 | 0.0433        |
| 0.6220 | 31500 | 0.0439        |
| 0.6319 | 32000 | 0.039         |
| 0.6417 | 32500 | 0.043         |
| 0.6516 | 33000 | 0.0352        |
| 0.6615 | 33500 | 0.0438        |
| 0.6714 | 34000 | 0.041         |
| 0.6812 | 34500 | 0.0445        |
| 0.6911 | 35000 | 0.0428        |
| 0.7010 | 35500 | 0.0445        |
| 0.7109 | 36000 | 0.0416        |
| 0.7207 | 36500 | 0.0448        |
| 0.7306 | 37000 | 0.0451        |
| 0.7405 | 37500 | 0.0392        |
| 0.7504 | 38000 | 0.0431        |
| 0.7602 | 38500 | 0.0402        |
| 0.7701 | 39000 | 0.0414        |
| 0.7800 | 39500 | 0.043         |
| 0.7898 | 40000 | 0.0406        |
| 0.7997 | 40500 | 0.0412        |
| 0.8096 | 41000 | 0.0415        |
| 0.8195 | 41500 | 0.0419        |
| 0.8293 | 42000 | 0.0397        |
| 0.8392 | 42500 | 0.0368        |
| 0.8491 | 43000 | 0.0454        |
| 0.8590 | 43500 | 0.0388        |
| 0.8688 | 44000 | 0.0393        |
| 0.8787 | 44500 | 0.0391        |
| 0.8886 | 45000 | 0.0412        |
| 0.8984 | 45500 | 0.0333        |
| 0.9083 | 46000 | 0.0472        |
| 0.9182 | 46500 | 0.0376        |
| 0.9281 | 47000 | 0.0447        |
| 0.9379 | 47500 | 0.0381        |
| 0.9478 | 48000 | 0.0378        |
| 0.9577 | 48500 | 0.0435        |
| 0.9676 | 49000 | 0.0414        |
| 0.9774 | 49500 | 0.0375        |
| 0.9873 | 50000 | 0.0399        |
| 0.9972 | 50500 | 0.0421        |

</details>

### Framework Versions
- Python: 3.12.11
- Sentence Transformers: 5.1.2
- Transformers: 4.57.1
- PyTorch: 2.9.0+cu128
- Accelerate: 1.12.0
- Datasets: 4.3.0
- Tokenizers: 0.22.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->