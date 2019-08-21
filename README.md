<h6>Object detection the project I worked, here is the article I wrote on the medium</h6>
<h6>Object segmentation</h6>
<h6>other image related such as deep dreams, neural style and old fashion way to experiment image</h6> 

I will first just write something I plan to write for the machine learning for SKL, Keras and TF here. 
so when join the machine learning party, all we need to do is learn from the data by using the production-ready frameworks. 

## Getting started

* [Set up article](https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9) 
* [Official tensorflow resource](https://github.com/tensorflow/models/tree/master/research/object_detection)   

## Setup to run the existing model result in repo
## Environment set up on GPU 
### AWS EC2 GPU
- Login the AWS and select EC2 pane
- Create an ssh key, scroll in the left menu find ‘Key pairs’ then click the ‘Import Key Pair’ button upload the ssh key
or ```cp .ssh/id_rsa.pub /mnt/c/Temp/ ```
- Search for ‘deep learning’ and select the first option: Deep Learning AMI (Ubuntu) Version (latest one) 
- Scroll down until you find ‘p3.xlarge’ and select it, 
- Upon the drop-down menu, select the YOUR SSH KEY and then press ‘Launch’ 

### Access the AWS instance via SSH 
```bash 
$ ssh -i ~/.ssh/<your_private_key_pair> -L localhost:8888:localhost:8888 ubuntu@<your instance IP> 
``` 
### Set up jupyter notebook and tensorboard 
first get clone the tensorflow models repo 
```bash
https://github.com/tensorflow/models 
```
then need to write this on command to initilize the tensorflow env and make sure the consistency of right kernal chosen in jupyter notebook 
```bash
$ source activate tensorflow_p36 
```
on the path command  
```bash 
$ ~/models/research
$ protoc object_detection/protos/*.proto --python_out=.
$ export
PYTHONPATH=$PYTHONPATH::/home/ubuntu/cocacola-201904/coke_dataset/models/research:/home/ubuntu/cocacola-201904/coke_dataset/models/research/slim 
```

## Detailed set up instruction(to do) 
<ul>
  <li>Git clone [Tensorflow Models module](https://github.com/tensorflow/models.git)
    
```bash
$ git clone https://github.com/tensorflow/models.git 
```
</li>
  <li>TFRecord</li>
  <li>LabelImg

```bash
$ git clone https://github.com/tzutalin/labelImg.git 
$ sudo apt-get install pyqt5-dev-tools
$ sudo pip3 install lxml
$ make qt5py3
```
  </li>
  <li>Start the image label program 
  
```bash 
$ python labelImg.py 
``` 
  </li> 
<li>Model Selection <a href='https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md'> model selection </a>
</li> 
<li>Model Training </li> 
<li>Model Inferenc export_inference_graph.py </li>
<li>model evualation mAP(mean Average Pricision), AP, Recall and Precision
</li>
<ul> 

## General Pipeline
<ul>
    <li>
        1. image collection and image dataset preprocessing (use Keras lib)
    </li>
    <li>
        2. model architecture configuration (use TF(transfer learning)) 
    </li>
    <li>
        3. model training (compare the GPU and CPU, GPU is 4.5 times faster than CPU, on GPU normally takes 2 hours)
    </li>
    <li>
        4. model inference (postprocessing, for object detection need to algorithm like soft NMS), on this part we first need to export the trained model graph, run pre-made file object_detection_tutorial.ipynb for simple sample image test, but for the time I was using I rewrote the code, and implement library boto to connect with AWS s3 for automatically read/download the file, implement the output code via csv format, besides on this step we need to use the image tag tool to get the groud truth class, so that can do model performance analysis to build the retrain pipeline loop, the three metric we used, confusion matrix, ROC AUC curve, and recall and precision 
    </li>
    <li>
        5. deploy to the sagemaker as an endpoint and use the api to directly connect to database for massive image usage
    </li>
</ul>

for image preprocessing, the tool used is LabelImg, resize the image to the certain pixel can slightly improve the model performance but limited. <br/> 

data formatting from xml-> csv-> TFRecord, three document need to use 
(you can find them on the first level of directory, but you can just select this and build for your own too, the files you need is xml_to_csv.py, split labels.ipynb, generate_tfrecord.py), all the generted training data train/validation.TFRecord data in to the YOUR/FOLDER/PATH/object_detection/data/ <br/> 

config 3 more different files before training, to config the model architecture path and train/validation path 
- faster_rcnn_resnet101_coco.config (or any other baseline models you prefer to use) in the training folder 
- change .pbtxt file, to implement all your training classes here in the JSON format in the object_detection/data/
- pipeline.config from object_detection/legacy/models/train/YOUR_MODEL_NAME (need download from <a href='https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)'>tensorflow model zoo site</a> <br/> 

start training for 20k steps and config the tensorboard and check the graph trend for model performance during the training  
first in the folder YOUR/FOLDER/PATH/models/research 
$ protoc object_detection/protos/*.proto --python_out=.
$ export
PYTHONPATH=$PYTHONPATH::YOUR/FOLDER/PATH/models/research:YOUR/FOLDER/PATH/models/research/slim 
config the training script on the folder path: YOUR/FOLDER/PATH/models/research/object_detection/legacy
example script: 
python train.py --train_dir=YOUR/FOLDER/PATH/models/research/object_detection/legacy/models/train --pipeline_config_path=YOUR/FOLDER/PATH/training/faster_rcnn_resnet101_coco.config 
training time for normally 2 hours in GPU but takes 10 more hours on CPU, check the tensorboard first to know the model performance and also to check if need to stop training to aviod overfitting problem 

model validation 
on the folder location: YOUR/FOLDER/PATH/models/research/object_detection/legacy 
example script: 
python eval.py --checkpoint_dir=YOUR/FOLDER/PATH/models/research/object_detection/legacy/models/train --eval_dir='eval' --pipeline_config_path=YOUR/FOLDER/PATH/training/faster_rcnn_resnet101_coco.config 

model inference use the re-created file on the object detection folder,first export the model trained result 
on the folder location: YOUR/FOLDER/PATH/models/research/object_detection/
example script: 
python export_inference_graph.py --input_type image_tensor --pipeline_config_path YOUR/FOLDER/PATH/training/faster_rcnn_resnet101_coco.config --trained_checkpoint_prefix legacy/models/train/model.ckpt-YOUR-TRAINING-STEPS--output_directory legacy/models/train 

then just config the saved_model path and class.pbtxt path on the inference code. 
for the mAP and recall, the higher the better model performance, normally mAP around 50 is a good one. 