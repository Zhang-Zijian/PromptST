FILE_OUTPUT="out_st_com19.csv"
i=0
while(($i<=18))
do
python main.py --pmt_flag --num_st_pmt 2 --basic_state_dict model_para/pretrained_c19.pt --dataset_name complaint19_3h${i} --device cuda:1 --epoch 2000 --out_dir $FILE_OUTPUT
let "i++"
done