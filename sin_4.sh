FILE_OUTPUT="out_sin_4.csv"
for i in {0..3}
do
python main.py --dataset_name nyctaxi2014_${i} --epoch 2000 --device cuda:1 --out_dir $FILE_OUTPUT
done 