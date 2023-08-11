FILE_OUTPUT="out_sin_19.csv"
for i in {0..18}
do
python main.py --dataset_name complaint19_3h${i} --epoch 2000 --device cuda:1 --out_dir $FILE_OUTPUT
done 