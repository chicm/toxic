for i in $1
do
    echo 'training bert-base-cased ifold:' $i
    python3 train.py --model_name bert-large-uncased --run base --batch_size 180 --use_path --num_epochs 2 --ifold $i --no_weight
    python3 train.py --model_name bert-large-uncased --run base --batch_size 180 --use_path --num_epochs 1 --ifold $i --lr 1e-5 --no_weight
done
