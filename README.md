Code for ACL2021 paper paper 'Multi-view Cross-Lingual Structured Prediction'.

## Sequence Labeling
#### Step 1: build config file
For example, label set and labeled data index.

`python build_config --config conll.config`

#### Step 2: train source models(optional)
One can also use pre-trained models.

`CUDA_VISIBLE_DEVICES=$gpu python train_source.py --target_language $target_language --target_name $target_name --language $language --name $name --config $target.config`

or,

`run run_source.sh $conll_file $gpu`

#### Step 3: train MuV models.
`CUDA_VISIBLE_DEVICES=$gpu nohup python train.py --aggregate_method $aggregate_method --source_name $source_name --language $language --name $name --config $target.config`

or,

`run run.sh $conll_file $gpu`
