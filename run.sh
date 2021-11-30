target=$1
gpu=$2


aggregate_method=att # aggregate multiple sources, [att, att_sent, avg_trainable, avg_equal]
source_name=en,de,nl # source model prefix
language=conll_03_spanish # target language, corpus
name=es # target language name, also source model suffix
#source_name=de,nl,es
#language=conll_03_english
#name=en

CUDA_VISIBLE_DEVICES=$gpu nohup python train.py \
    --aggregate_method $aggregate_method \
    --source_name $source_name \
    --language $language \
    --name $name \
    --config $target.config > logs/CoNLL/MuV/${name}_att.log1 &
echo "CUDA_VISIBLE_DEVICES=$gpu nohup python train.py
    --aggregate_method $aggregate_method
    --source_name $source_name
    --language $language
    --name $name
    --config $target.config > logs/CoNLL/MuV/${name}_att.log1 &"
