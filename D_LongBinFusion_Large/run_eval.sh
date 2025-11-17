for i in {10..19}
do
    echo "Evaluating with weight $i ..."
    python -W ignore eval.py \
        --config_file=./configs/ddad/test_version.yaml \
        --weight_path=./lightweight_longbinfusion_ver1/test_version/models/weights_${i}/ \
        > logs_eval/weight__${i}.txt 2>&1
done
