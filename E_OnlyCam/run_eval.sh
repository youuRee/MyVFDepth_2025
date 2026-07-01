for i in {10..19}
do
    echo "Evaluating with weight $i ..."
    python3 -W ignore eval.py \
        --config_file=./configs/ddad/ddad_surround_fusion.yaml \
        --weight_path=./results/ddad_surround_fusion/models/weights_${i}/ \
        > logs_eval/weight__${i}.txt 2>&1
done
