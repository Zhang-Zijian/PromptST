FILE_OUTPUT="out_st_taxi.csv"
i=0
while(($i<=3))
do
python main.py --pmt_flag --num_st_pmt 2 --basic_state_dict model_para/pretrained_taxi.pt --dataset_name nyctaxi2014_${i} --device cuda:1 --epoch 2000 --learning_rate 0.001 --out_dir $FILE_OUTPUT
let "i++"
done