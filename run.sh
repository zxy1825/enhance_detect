#!/bin/bash
###
 # @Author: gw00336465 gw00336465@ifyou.com
 # @Date: 2024-03-28 13:05:21
 # @LastEditors: gw00336465 gw00336465@ifyou.com
 # @LastEditTime: 2024-04-12 14:04:22
 # @FilePath: /YOLOv5/operate.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 

# 显示操作提示信息
echo "Which operation will be operated?"
echo "1. train"
echo "2. val"
echo "3. detect"

# 接受用户输入
read -p "Your choice:" choice

# 使用 case 语句执行相应分支语句
case $choice in
    1)
        # 执行操作一的语句
        echo "Start to train on 5 ExDark datasets, basic models is yolov5s"
        # python -m torch.distributed.run --nproc_per_node 2 train.py --img 640 --batch 128 --epochs 50 --data ./data/Exdark_Light.yaml --cfg ./models/yolov5s.yaml --weights yolov5s.pt --device 0,1
        python -m torch.distributed.run --nproc_per_node 2 train.py --img 640 --batch 128 --epochs 150 --data ./data/Exdark_Source.yaml --cfg ./models/yolov5s.yaml --weights ./runs/train/COCO_s/weights/best.pt --device 0,1
        python -m torch.distributed.run --nproc_per_node 2 train.py --img 640 --batch 128 --epochs 150 --data ./data/Exdark_Gamma.yaml --cfg ./models/yolov5s.yaml --weights ./runs/train/COCO_s/weights/best.pt --device 0,1
        # python -m torch.distributed.run --nproc_per_node 2 train.py --img 640 --batch 32 --epochs 80 --data ./data/Exdark_Kind.yaml --cfg ./models/yolov5x.yaml --weights ./runs/train/COCO/weights/best.pt --device 0,1
        python -m torch.distributed.run --nproc_per_node 2 train.py --img 640 --batch 128 --epochs 150 --data ./data/Exdark_MEBBLN.yaml --cfg ./models/yolov5s.yaml --weights ./runs/train/COCO_s/weights/best.pt --device 0,1
        python -m torch.distributed.run --nproc_per_node 2 train.py --img 640 --batch 128 --epochs 150 --data ./data/Exdark_ZeroDCE.yaml --cfg ./models/yolov5s.yaml --weights ./runs/train/COCO_s/weights/best.pt --device 0,1
        ;;
    2)
        # 执行操作二的语句
        echo "Start to val on 5 ExDark datasets, basic models is yolov5l6"
        # 在这里添加操作二的具体代码
        python val.py --weight runs/train/Source3/weights/best.pt --data data/Exdark_Source.yaml
        python val.py --weight runs/train/Gamma3/weights/best.pt --data data/Exdark_Gamma.yaml
        python val.py --weight runs/train/MEBBLN3/weights/best.pt --data data/Exdark_MEBBLN.yaml
        python val.py --weight runs/train/ZeroDCE3/weights/best.pt --data data/Exdark_ZeroDCE.yaml
        ;;
    3)
        # 执行操作三的语句
        echo "您选择了操作三"
        # 在这里添加操作三的具体代码
        ;;
    *)
        # 处理无效输入的情况
        echo "无效的选项，请重新输入"
        ;;
esac

