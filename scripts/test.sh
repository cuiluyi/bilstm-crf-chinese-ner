export CUDA_VISIBLE_DEVICES=0

python test.py --model_path ckpts/model_epoch_1.pth \
> logs/test_epoch_1.log 2>&1 &

python test.py --model_path ckpts/model_epoch_2.pth \
> logs/test_epoch_2.log 2>&1 &

python test.py --model_path ckpts/model_epoch_3.pth \
> logs/test_epoch_3.log 2>&1 &

python test.py --model_path ckpts/model_epoch_4.pth \
> logs/test_epoch_4.log 2>&1 &

python test.py --model_path ckpts/model_epoch_5.pth \
> logs/test_epoch_5.log 2>&1 &

python test.py --model_path ckpts/model_epoch_6.pth \
> logs/test_epoch_6.log 2>&1 &

python test.py --model_path ckpts/model_epoch_7.pth \
> logs/test_epoch_7.log 2>&1 &

python test.py --model_path ckpts/model_epoch_8.pth \
> logs/test_epoch_8.log 2>&1 &

python test.py --model_path ckpts/model_epoch_9.pth \
> logs/test_epoch_9.log 2>&1 &

python test.py --model_path ckpts/model_epoch_10.pth \
> logs/test_epoch_10.log 2>&1 &
