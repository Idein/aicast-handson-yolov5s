# aicast-handson-yolov5s app

ai castハンズオン用に作成したアプリです。
aicastでyolov5s物体検出を行います。

<img src=./screenshot.jpg>

## build
```
cp ../precompiled/yolov5s.hef ./app/
make
actdk build
```

## run
```
actdk remote add aicast@192.168.xx.xx
actdk run aicast
```

