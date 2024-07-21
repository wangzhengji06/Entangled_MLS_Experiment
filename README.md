This repo is for the Entangled MLS Experiment. 

# 3d object detection system experiment

The experiments are located in the 3d_object_detection_experiment folder. 

## Data Preparation

Download the KITTI Dataset for 3d object detection from official website: https://www.cvlibs.net/datasets/kitti/  

After downloading the dataset, organize it in the following way:

    ./Kitti/object
    
        train.txt
        val.txt
        test.txt 
    
        training/
            calib/
            image_2/ #left image
            image_3/ #right image
            label_2/
            velodyne/ 
    
        testing/
            calib/
            image_2/
            image_3/
            velodyne/



## Upstream Model: PSMNET

We mainly follow the process at the repo: https://github.com/facebookresearch/self_defeating_improvements/tree/main/pseudo_lidar_exp  as well as the repo: https://github.com/mileyan/pseudo_lidar. The environment setup is exactly the same as the above 2 repos. 

### Generate ground-truth disparity

```
cd ./preprocessing
python generate_disp.py --data_path {your_path/KITTI/object/training/} --split_file {your_path/KITTI/object/trainval.txt}
```

### Train the Disparity Predictor

The training follows the same process at the repo: https://github.com/JiaRenChang/PSMNet

#### Train the model under disparity L1 loss

```
cd ./psmnet
mkdir kitti_3d
python finetune_3d.py --maxdisp 192 --model stackhourglass --datapath {your_path/Kitti/object/training/} --split_file {your_path/Kitti/object/train.txt}  --epochs 300 --lr_scale 50 --loadmodel ./pretrained_sceneflow.tar --savemodel ./psmnet/kitti_3d/  --btrain 12
```

#### Train the model under depth L1 loss

```
cd ./psmnet
mkdir kitti_3d_dl
python finetune_3d.py --maxdisp 192 --model stackhourglass --datapath {your_path/Kitti/object/training/} --split_file {your_path/Kitti/object/train.txt}  --epochs 300 --lr_scale 50 --loadmodel ./pretrained_sceneflow.tar --savemodel ./kitti_3d_dl/  --btrain 12 --data_type depth
```

### Generate the Pseudo-lidar

#### Generate the point cloud using disparity loss

```
cd ./psmnet
python submission.py \
    --loadmodel ./kitti_3d/finetune_300.tar \
    --datapath {your_path/KITTI/object/training/} \
    --save_path {your_path/KITTI/object/training/predict_disparity}   
python ./generate_lidar.py  \
    --calib_dir {your_path/KITTI/object/training/calib/} \
    --save_dir {your_path/KITTI/object/training/pseudo-lidar_velodyne/} \
    --disparity_dir {your_path/KITTI/object/training/predict_disparity} \
    --max_high 1
cd ..
cd ./preprocessing
python /kitti_sparsify.py --pl_path  ../Kitti/object/training/pseudo-lidar_velodyne --sparse_pl_path  ~/Kitti/object/training/pseudo-lidar_velodyne_sparse/

```



#### Generate the point cloud using depth loss

```
cd ./psmnet
python submission.py \
    --loadmodel ./kitti_3d_dl/finetune_300.tar \
    --datapath {your_path/KITTI/object/training/} \
    --save_path {your_path/KITTI/object/training/predict_disparity}   
python ./generate_lidar.py  \
    --calib_dir {your_path/KITTI/object/training/calib/} \
    --save_dir {your_path/KITTI/object/training/pseudo-lidar_velodyne_dl/} \
    --disparity_dir {your_path/KITTI/object/training/predict_disparity_dl} \
    --max_high 1
cd ..
cd ./preprocessing
python /kitti_sparsify.py --pl_path  ../Kitti/object/training/pseudo-lidar_velodyne_dl --sparse_pl_path  ~/Kitti/object/training/pseudo-lidar_velodyne_dl_sparse/
```

Currently you should have two versions of pseudo-lidar point clouds. One is in `Kitti/object/training/pseudo-lidar_velodyne_dl_sparse/` and one is in `Kitti/object/training/pseudo-lidar_velodyne_sparse/`.

## Downstream Model: Point-RCNN

We mainly follow the process at the repo: https://github.com/open-mmlab/OpenPCDet

### Clone OpenPCDet

```
git clone https://github.com/open-mmlab/OpenPCDet.git
```

### Prepare 2 versions of Dataset with different point clouds

```
OpenPCDet
├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2
├── pcdet
├── tools
```

Replace the velodyne with 2 versions of point clouds we get form the above.

### Train and  Validate Using Point-RCNN

Use the same command for different versions of dataset: 

```
# Preoprocess the data
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
# Training
python train.py --cfg_file cfgs/kitti_models/pointrcnn.yaml --batch_size=8 
# Test
python test.py --cfg_file cfgs/kitti_models/pointrcnn.yaml --batch_size=8  --ckpt /path/checkpount.pth
```

The trained model is provided as `depth_point_rcnn.pth` and `disparity_point_rcnn.pth`. 



# sentence classification experiment

The dataset can be downloaded via this link: https://file.io/khLxVpXur2n3

The environment setup is in requiremts.txt. 

Please run the following command to recreate the experiment result:

```
nohup python textcnn_train.py > textcnn.log &
nohup python bert_model.py > bert.log &
nohup python stacking.py > all.log &
```

Then check the result in `textcnn.log`,`bert.log` and `all.log` . `textcnn.log` records the upstream model text CNN's result on dataset.   `bert.log` records the upstream model BERT's result on dataset. `all.log` would provide the upstream + downstream model's result for both text CNN and BERT. 
