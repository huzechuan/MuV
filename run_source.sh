target=$1
gpu=$2

target_language=conll_03_english,conll_03_german,conll_03_spanish # target language, corpus
target_name=en,de,es # target name, also source model suffix
language=conll_03_dutch # source language, corpus
name=nl # source name, also source model prefix

CUDA_VISIBLE_DEVICES=$gpu nohup python train_source.py \
    --target_language $target_language \
    --target_name $target_name \
    --language $language \
    --name $name \
    --config $target.config > logs/CoNLL/Direct/$name.log &
echo "CUDA_VISIBLE_DEVICES=$gpu nohup python train_source.py
    --target_language $target_language
    --target_name $target_name
    --language $language
    --name $name
    --config $target.config > logs/CoNLL/Direct/$name.log &"
