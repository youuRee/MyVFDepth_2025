for i in {0..19}
do
    echo "Evaluating with weight $i ..."
    python -W ignore eval.py \
        --config_file=./configs/ddad/ddad_surround_fusion_augdepth.yaml \
        --weight_path=./gtpose_synth/ddad_surround_fusion_augdepth/models/weights_${i}/ \
        > logs_eval/weight__${i}.txt 2>&1
done