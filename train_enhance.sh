time=$(date "+%Y.%m.%d-%H.%M.%S")
echo run-time is $time

export PYTHONPATH=$(pwd)
python enhance/train.py -e 40 -b 2 -d '/mnt/data/ExDark/Source_ExDark/images/val' -m -a 0.84 -c 3 --bilinear -v 10.0  > logs/log_$time.txt & 