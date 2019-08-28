## semantic segmentation 

Concept: 
Semantic segmentation is understanding an image at pixel level, to assign semantic labels to every pixel to a class. which is one step close to instance segmentation. For senmantic segmentation, one of the architectures is the FCN (Fully Convolutional Network), by Long et al. from Berkeley, in 2014, popularized CNN architectures for dense predictions without any fully connected layers. This allowed segmentation maps to be generated for image of any size and was also much faster compared to the patch classification approach. Almost all the subsequent state of the art approaches on semantic segmentation adopted this paradigm. 
another architecture is from Google DeepLabv3 (<a href='https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html?m=1'>link</a>), for the faster and stronger encoder-decoder network for semantic segmentation. 
<img src='https://2.bp.blogspot.com/-gxnbZ9w2Dro/WqMOQTJ_zzI/AAAAAAAACeA/dyLgkY5TnFEf2j6jyXDXIDWj_wrbHhteQCLcBGAs/s640/image2.png'> 

The most common arch is from encoder to decoder, from downsampling to the upsampling. 
Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation (<a href='https://arxiv.org/abs/1802.02611'>arxiv link</a>)

test with pixel value for each class, for each class will have certain range of pixel to define which class it belong to. make decision for every pixel of each image,So, for each pixel, the model needs to classify it as one of the pre-determined classes.<br/> 

There are two common ways to do downsampling in neural nets. By using convolution striding or regular pooling operations. In general, downsampling has one goal. To reduce the spatial dimensions of given feature maps. For that reason, downsampling allows us to perform deeper convolutions without much memory concerns. Yet, they do it in detriment of losing some features in the process. <br/> 

Deeplab uses an ImageNet pre-trained ResNet as its main feature extractor network.
- ResNet architecture 
One of the main contributions of ResNets was to provide a framework to ease the training of deeper models. 
The two 1x1 operations are designed for reducing and restoring dimensions, bottleneck units are more suitable for training deeper models because of less training time and computational resources need  
- Atrous Spatial Pyramid Pooling (ASPP) 
- Global Average Pooling (GAP) 

Resource: 
- http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review#fcn 
- http://people.inf.ethz.ch/aksoyy/sss/ 
