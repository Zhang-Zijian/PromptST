FILE_OUTPUT="out_ft_taxi.csv"
i=0
while(($i<=3))
do 
cp model_para/pretrained_taxi.pt model_para/pretrained_taxi_${i}.pt &&
python main.py --resume_dir model_para/pretrained_taxi_${i}.pt --dataset_name nyctaxi2014_${i} --epoch 2000 --early_stop 100 --ft_flag --device cuda:2 --out_dir $FILE_OUTPUT
let "i++"
done
