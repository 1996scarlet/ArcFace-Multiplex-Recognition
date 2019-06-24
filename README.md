
# CV Object Detection and Recognition

## How to build Darknet and R-CNN

* `chmod a+x ./build_darknet_and_rcnn.sh`
* `./build_darknet_and_rcnn.sh`

## How to train mlp classifier

* `mkdir ./Temp/raw`
* `mkdir ./Temp/train_data`
* Place training pictures in the following format：

    ```shell
    ─── train_data
        ├── bush
        │   ├── 1559637960.1595788.jpg
        │   ├── 1559637960.1762984.jpg
        │   └── 1559637960.2001894.jpg
        ├── clinton
        │   ├── 1559637960.2104468.jpg
        │   ├── 1559637960.2225769.jpg
        │   └── 1559637960.281161.jpg
        └── obama
            ├── 1559637960.2940397.jpg
            ├── 1559637960.31212.jpg
            └── 1559637960.3381834.jpg
    ```

* `python3 train_mlp.py`

## How to run

* Node-JS Server `node ../NodeServer/server.js`
* CDS `python3 peropero_cds_v3.py` at `http://127.0.0.1:6789/cds`
* TOOLS `python3 niconico_tools.py` at `http://127.0.0.1:6789/tools`
* LUNATIC `python3 mikomiko_hk_v4.py` at `http://127.0.0.1:6789/lunatic`
