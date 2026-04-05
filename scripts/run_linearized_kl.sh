datasets=("polaris" "math")
model_names=

for dataset in ${datasets[@]}; do
    python analysis/get_linearized_kl.py --data $dataset

done
done