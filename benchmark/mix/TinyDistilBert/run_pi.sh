num_repeat=5
hardware="pi"
savefile="mix.csv"
python run.py $hardware False $savefile
for((i=1;i<$num_repeat;i++))
do
    python run.py $hardware False $savefile
done
