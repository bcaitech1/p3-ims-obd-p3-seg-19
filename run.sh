#!/bin/bash

## 파일 위치: code/mmdetection_trash
## 파일 실행 방법: bash run.sh

## 기본 코드
# python tools/train.py configs/trash/faster_rcnn/faster_rcnn_r50_fpn_1x_trash.py
# python tools/test.py configs/trash/faster_rcnn/faster_rcnn_r50_fpn_1x_trash.py work_dirs/faster_rcnn_r50_fpn_1x_trash/epoch_12.pth --out work_dirs/faster_rcnn_r50_fpn_1x_trash/epoch_12.pkl
# python pkl_to_submission.py --pkl work_dirs/faster_rcnn_r50_fpn_1x_trash/epoch_12.pkl --csv submission.csv

# 함수
function run(){
    # 인자
    CONFIG=$1 # config파일 (폴더/파일명.py)
    FOLDER=$2 # 저장할 폴더 이름 (마음대로 설정 가능)
    MODEL=$3 # test에 사용할 모델 이름
    SEED=${4:-19}  # 랜덤시드 (default값: 19)
    
    python tools/train.py configs/trash/${CONFIG} --work-dir work_dirs/${FOLDER} --seed ${SEED} --deterministic
    python tools/test.py configs/trash/${CONFIG} work_dirs/${FOLDER}/${MODEL}.pth --out work_dirs/${FOLDER}/${MODEL}.pkl
    python pkl_to_submission.py --pkl work_dirs/${FOLDER}/${MODEL}.pkl --csv work_dirs/${FOLDER}/submission.csv
    python autosubmit.py --folder ${FOLDER}
}

## 함수 실행: run [config파일] [저장폴더] [모델이름] (SEED)
## 예시
# run "faster_rcnn/faster_rcnn_r50_fpn_1x_trash.py" "faster_rcnn_r50_fpn_1x_trash" "epoch_12" 42
# run "detectors/detectors_r50_1x_trash.py" "detectors_r50" "epoch_12"

# run "faster_rcnn/faster_rcnn_r50_fpn_1x_trash.py" "faster_rcnn_r50_fpn_SoftNMS" "epoch_12"
# run "detectors/detectors_r50_1x_trash.py" "detectors_r50_SoftNMS" "epoch_12"
# run "detectors/detectors_resnext101_1x_trash.py" "detectors_resnext101" "epoch_12"
# run "faster_rcnn/faster_rcnn_r50_fpn_1x_trash.py" "faster_rcnn_r50_fpn_RandomRotate30" "epoch_12"
# run "faster_rcnn/faster_rcnn_r50_bifpn_1x_trash.py" "faster_rcnn_r50_bifpn_RandomRotate30" "epoch_12"
# run "detectors/detectors_r101_1x_trash.py" "detectors_r101_RandomRotate30" "epoch_12"
run "yolo/swin_tiny.py" "swin_tiny_pretrained" "best_bbox_mAP_50"