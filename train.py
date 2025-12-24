
from ultralytics import YOLO



if __name__ == '__main__':
    # 加载预训练模型
    model = YOLO('ultralytics/cfg/models/v8.0/GF-YOLO.yaml')

    # 训练模型
    model.train(
        data="E://python_program//visdrone_yolo//VisDrone2019.yaml",  # 数据集配置
        epochs=200,
        batch=16,
        imgsz=640,

        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        warmup_epochs=3.0,
        close_mosaic=190,
        save_period=50,
        plots=True,
        verbose=True,
        cache=True,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        optimizer='SGD',
        device=0,
        workers=16,
    )


