FILE_OUTPUT="out_ft_com19.csv"
i=0
while(($i<=18))
do 
cp model_para/pretrained_c19.pt model_para/pretrained_c19_${i}.pt &&
python main.py --ft_flag --resume_dir model_para/pretrained_c19_${i}.pt --dataset_name complaint19_3h${i} --epoch 2000 --learning_rate 0.0001 --device cuda:0 --out_dir $FILE_OUTPUT
let "i++"
done
