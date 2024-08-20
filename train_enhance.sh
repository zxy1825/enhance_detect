time=$(date "+%Y.%m.%d-%H.%M.%S")
echo run-time is $time

export PYTHONPATH=$(pwd)
python enhance/train.py -e 100 -b 2 -d '/mnt/data/ExDark/Source_ExDark/images/all' -m -a 0.8 -c 3 --bilinear -v 10.0  > logs/log_$time.txt & 