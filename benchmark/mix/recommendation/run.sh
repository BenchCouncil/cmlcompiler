num_repeat=5
hardware="cpu"
savefile="mix.csv"
python run.py $hardware True $savefile
for((i=1;i<$num_repeat;i++))
do
    python run.py $hardware False $savefile
done
