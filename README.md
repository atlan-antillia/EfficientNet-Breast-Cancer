<h2>EfficientNet-Breast-Cancer (Updated: 2022/08/24)</h2>
<a href="#1">1 EfficientNetV2 Breast Cancer BreaKHis Classification </a><br>
<a href="#1.1">1.1 Clone repository</a><br>
<a href="#1.2">1.2 Prepare Breast Cancer dataset</a><br>
<a href="#1.3">1.3 Install Python packages</a><br>
<a href="#2">2 Python classes for Breast Cancer Classification</a><br>
<a href="#3">3 Pretrained model</a><br>
<a href="#4">4 Train</a><br>
<a href="#4.1">4.1 Train script</a><br>
<a href="#4.2">4.2 Training result</a><br>
<a href="#5">5 Inference</a><br>
<a href="#5.1">5.1 Inference script</a><br>
<a href="#5.2">5.2 Sample test images</a><br>
<a href="#5.3">5.3 Inference result</a><br>
<a href="#6">6 Evaluation</a><br>
<a href="#6.1">6.1 Evaluation script</a><br>
<a href="#6.2">6.2 Evaluation result</a><br>

<h2>
<a id="1">1 EfficientNetV2 Breat Cancer BreaKHis Classification</a>
</h2>

 This is an experimental Breast Cancer BreaKHis_V1_400X Classification project based on <b>efficientnetv2</b> in <a href="https://github.com/google/automl">Brain AutoML</a>.
<br>
 The original Breast Cancer dataset BreakHis_v1 has been taken from the following web site:<br>
<b>Laboratório Visão Robótica e Imagem</b>
<br>
<a href="https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/">
Breast Cancer Histopathological Database (BreakHis)
</a>
<br>
<br>We use python 3.8 and tensorflow 2.8.0 environment on Windows 11.<br>
<h3>
<a id="1.1">1.1 Clone repository</a>
</h3>
 Please run the following command in your working directory:<br>
<pre>
git clone https://github.com/atlan-antillia/EfficientNet-Skin-Cancer.git
</pre>
You will have the following directory tree:<br>
<pre>
.
├─asset
└─projects
    └─Breast-Cancer
        ├─eval
        ├─evaluation
        ├─inference        
        └─test
</pre>
<h3>
<a id="1.2">1.2 Prepare Breast Cancer dataset</a>
</h3>

Please download the dataset <b>BreakHis_v1</b> from the following web site:
<br>
<a href="https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/"><b>Breast Cancer Histopathological Database (BreakHis)</b></a>
<br>
<br>
The original images (700X460 pixels, 3-channel RGB, 8-bit depth in each channel, PNG format) of <b>benign</b> and <b>malignant</b> in <b>BreakHis_v1</b> 
are stored in the different folders specified by magnifying factors (40X, 100X, 200X and 400X) as shown below:<br>
<img src="./asset/BreaKHis_v1_histology_slides_breast_malignant_SOB_papillary_carcinoma_SOB_M_PC_14-9146_400X.png" 
width="840" height="auto"><br>
<br>
For simplicity, we have selected the images in all 400X folders only, and created
<b> BreaKHis_V1_400X</b> dataset which contains <b>test</b> and <b>train</b>.<br> 

<pre>
.
├─asset
└─projects
    └─Breast-Cancer
        ├─BreaKHis_V1_400X
        │  ├─test
        │  │  ├─benign
        │  │  └─malignant
        │  └─train
        │      ├─benign
        │      └─malignant
        └─test     
　...
</pre>
If you would like to create <b>BeaKHis_V1_400X/master</b> dataset from 
the original dataset BreaKHis_v1, please move to ./projects/Breast-Cancer/ directory
, and run the following script:<br>
<pre>
>python create_BreaKHis_X400_master.py
</pre>
Also, you can easily create train and test dataset from the generated master dataset by using 
<b>split-folders</b> package.
<br><br>
Sample images of BreaKHis_V1_400X/malignant:<br>
<img src="./asset/Brest_Cancer_BreaKHis_V1_400X_malignant.png" width="840" height="auto">
<br> 
<br>
Sample images of BreaKHis_V1_400X/benign:<br>
<img src="./asset/Brest_Cancer_BreaKHis_V1_400X_benign.png" width="840" height="auto">
<br> 
<br>

The number of images in train dataset:<br>
<img src="./asset/BreaKHis_V1_400X_train.png" width="540" height="auto">
<br>
<br>
The number of images in test dataset:<br>
<img src="./asset/BreaKHis_V1_400X_test.png" width="540" height="auto">
<br>
<br>

<h3>
<a id="#1.3">1.3 Install Python packages</a>
</h3>
Please run the following commnad to install Python packages for this project.<br>
<pre>
pip install -r requirements.txt
</pre>
<br>

<h2>
<a id="2">2 Python classes for Breast Cancer Classification</a>
</h2>
We have defined the following python classes to implement our Breast Cancer Classification.<br>
<li>
<a href="./ClassificationReportWriter.py">ClassificationReportWriter</a>
</li>
<li>
<a href="./ConfusionMatrix.py">ConfusionMatrix</a>
</li>
<li>
<a href="./CustomDataset.py">CustomDataset</a>
</li>
<li>
<a href="./EpochChangeCallback.py">EpochChangeCallback</a>
</li>
<li>
<a href="./EfficientNetV2Evaluator.py">EfficientNetV2Evaluator</a>
</li>
<li>
<a href="./EfficientNetV2Inferencer.py">EfficientNetV2Inferencer</a>
</li>
<li>
<a href="./EfficientNetV2ModelTrainer.py">EfficientNetV2ModelTrainer</a>
</li>
<li>
<a href="./FineTuningModel.py">FineTuningModel</a>
</li>

<li>
<a href="./TestDataset.py">TestDataset</a>
</li>

<h2>
<a id="3">3 Pretrained model</a>
</h2>
 We have used pretrained <b>efficientnetv2-m</b> to train Breast Cancer FineTuning Model.
Please download the pretrained checkpoint file from <a href="https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-m.tgz">efficientnetv2-m.tgz</a>, expand it, and place the model under our top repository.

<pre>
.
├─asset
├─efficientnetv2-m
└─projects
        ├─BreaKHis_V1_400X
  ...
</pre>

<h2>
<a id="4">4 Train</a>

</h2>
<h3>
<a id="4.1">4.1 Train script</a>
</h3>
Please run the following bat file to train our Breast Cancer efficientnetv2 model by using
<b>BreaKHis_V1_400X/train</b>.
<pre>
./1_train.bat
</pre>
<pre>
rem 1_train.bat
python ../../EfficientNetV2ModelTrainer.py ^
  --model_dir=./models ^
  --eval_dir=./eval ^
  --model_name=efficientnetv2-m  ^
  --data_generator_config=./data_generator.config ^
  --ckpt_dir=../../efficientnetv2-m/model ^
  --optimizer=rmsprop ^
  --image_size=384 ^
  --eval_image_size=480 ^
  --data_dir=./BreaKHis_V1_400X/train ^
  --data_augmentation=True ^
  --valid_data_augmentation=True ^
  --fine_tuning=True ^
  --monitor=val_loss ^
  --learning_rate=0.0001 ^
  --trainable_layers_ratio=0.4 ^
  --dropout_rate=0.3 ^
  --num_epochs=50 ^
  --batch_size=4 ^
  --patience=10 ^
  --debug=True  
</pre>
, where data_generator.config is the following:<br>
<pre>
; data_generation.config

[training]
validation_split   = 0.2
featurewise_center = True
samplewise_center  = False
featurewise_std_normalization=True
samplewise_std_normalization =False
zca_whitening                =False
rotation_range     = 60
horizontal_flip    = True
vertical_flip      = True
 
width_shift_range  = 0.1
height_shift_range = 0.1
shear_range        = 0.01
zoom_range         = [0.2, 2.0]
;zoom_range         = 0.2
data_format        = "channels_last"

[validation]8
validation_split   = 0.2
featurewise_center = True
samplewise_center  = False
featurewise_std_normalization=True
samplewise_std_normalization =False
zca_whitening                =False
rotation_range     = 60
horizontal_flip    = True
vertical_flip      = True
width_shift_range  = 0.1
height_shift_range = 0.1
shear_range        = 0.01
zoom_range         = [0.3, 2.0]
;zoom_range         = 0.1
data_format        = "channels_last"
</pre>

<h3>
<a id="4.2">4.2 Training result</a>
</h3>

This will generate a <b>best_model.h5</b> in the models folder specified by --model_dir parameter.<br>
Furthermore, it will generate a <a href="./projects/Breast-Cancer/eval/train_accuracies.csv">train_accuracies</a>
and <a href="./projects/Breast-Cancer/eval/train_losses.csv">train_losses</a> files
<br>
Training console output:<br>
<img src="./asset/Breast_Cancer_train_console_output_at_epoch_25_0824.png" width="740" height="auto"><br>
<br>
Train_accuracies:<br>
<!--
<img src="./asset/Breat_Cancer_accuracies_at_epoch_25_0824.png" width="740" height="auto"><br>
-->
<img src="./projects/Breast-Cancer/eval/train_accuracies.png" width="640" height="auto"><br>

<br>
Train_losses:<br>
<!--
<img src="./asset/Breast_Cancer_train_losses_at_epoch_25_0824.png" width="740" height="auto"><br>
-->
<img src="./projects/Breast-Cancer/eval/train_losses.png" width="640" height="auto"><br>

<br>
<h2>
<a id="5">5 Inference</a>
</h2>
<h3>
<a id="5.1">5.1 Inference script</a>
</h3>
Please run the following bat file to infer the skin cancer lesions in test images by the model generated by the above train command.<br>
<pre>
./2_inference.bat
</pre>
<pre>
rem 2_inference.bat
python ../../EfficientNetV2Inferencer.py ^
  --model_name=efficientnetv2-m  ^
  --model_dir=./models ^
  --fine_tuning=True ^
  --trainable_layers_ratio=0.4 ^
  --dropout_rate=0.3 ^
  --image_path=./test/*.png ^
  --eval_image_size=480 ^
  --label_map=./label_map.txt ^
  --mixed_precision=True ^
  --infer_dir=./inference ^
  --debug=False 
</pre>
<br>
label_map.txt:
<pre>
benign
malignant
</pre>
<br>
<h3>
<a id="5.2">5.2 Sample test images</a>
</h3>

Sample test images generated by <a href="./projects/Breast-Cancer/create_test_dataset.py">create_test_dataset.py</a> 
from <a href="./projects/Breast-Cancer/BreaKHis_V1_400X/test">BreaKHis_V1_400X/test</a>.
<br>
<img src="./asset/Breast_Cancer_BreaKHis_V1_400X_test.png" width="840" height="auto"><br>


<br>
<h3>
<a id="5.3">5.3 Inference result</a>
</h3>
This inference command will generate <a href="./projects/Breast-Cancer/inference/inference.csv">inference result file</a>.
<br>At this time, you can see the inference accuracy for the test dataset by our trained model is very low.
More experiments will be needed to improve accuracy.<br>

<br>
Inference console output:<br>
<img src="./asset/Breast_Cancer_infer_console_output_at_epoch_25_0824.png" width="740" height="auto"><br>
<br>

Inference result (inference.csv):<br>
<img src="./asset/Breast_Cancer_inference_at_epoch_25_0824.png" width="740" height="auto"><br>
<br>
<h2>
<a id="6">6 Evaluation</a>
</h2>
<h3>
<a id="6.1">6.1 Evaluation script</a>
</h3>
Please run the following bat file to evaluate <a href="./projects/Breast-Cancer/BreaKHis_V1_400X/test">
BreaKHis_V1_400X/test</a> by the trained model.<br>
<pre>
./3_evaluate.bat
</pre>
<pre>
rem 3_evaluate.bat
python ../../EfficientNetV2Evaluator.py ^
  --model_name=efficientnetv2-m  ^
  --model_dir=./models ^
  --data_dir=./BreaKHis_V1_400X/test ^
  --evaluation_dir=./evaluation ^
  --fine_tuning=True ^
  --trainable_layers_ratio=0.4 ^
  --dropout_rate=0.3 ^
  --eval_image_size=480 ^
  --mixed_precision=True ^
  --debug=False 
</pre>
<br>

<h3>
<a id="6.2">6.2 Evaluation result</a>
</h3>

This evaluation command will generate <a href="./projects/Breast-Cancer/evaluation/classification_report.csv">a classification report</a>
 and <a href="./projects/Breast-Cancer/evaluation/confusion_matrix.png">a confusion_matrix</a>.
<br>
<br>
Evaluation console output:<br>
<img src="./asset/Breast_Cancer_evaluate_console_output_at_epoch_25_0824.png" width="740" height="auto"><br>
<br>

<br>
Classification report:<br>
<img src="./asset/Breast_Cancer_classification_report_at_epoch_25_0824.png" width="740" height="auto"><br>
<br>
Confusion matrix:<br>
<img src="./projects/Breast-Cancer/evaluation/confusion_matrix.png" width="740" height="auto"><br>

