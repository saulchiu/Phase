Attack Baselines: **BadNets, Blend, WaNet, FTrojan, FIBA, DUBA**

Defense Baselines: **GradCam, NC, FIP, RNP, FIP, CLP, FP**

Train a backdoored model from stratch:
```shell
cd run
python train.py attack=phase dataset_name=cifar10 model=resnet18
```

Eval the ACC, ASR of the trained model:
```shell
cd run
python eval_acc.py --path /home/chengyiqiu/code/INBA/results/cifar10/inba/20241124165116_robust_training
```

Eval the SSIM... of the trained model:
```shell
cd run
python eval_ssim.py --path /home/chengyiqiu/code/INBA/results/cifar10/inba/20241124165116_robust_training
```

Test NC on your trained model:
```shell
cd defense/NC
python nc.py --path /home/chengyiqiu/code/INBA/results/cifar10/phase/resnet18/20241204184214 
```

# TO-Do
- Release the config details of training: Yes.
- Release the pre-train model: No
- Tiny the code: No
