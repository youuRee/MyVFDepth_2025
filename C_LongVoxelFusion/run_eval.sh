for i in {10..19}
do
    echo "Evaluating with weight $i ..."
    python3 -W ignore eval.py \
        --config_file=./configs/ddad/test_version.yaml \
        --weight_path=./long_voxel_cl_fusion/test_version/models/weights_${i}/ \
        > ./long_voxel_cl_fusion/test_version/logs_eval/weight_${i}.txt 2>&1
done