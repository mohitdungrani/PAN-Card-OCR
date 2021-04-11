# PAN-Card-OCR
## Introcution
PAN Card is one type of legal document The that is allocated by the income tax department to all the taxpayers in India.

The idea of this project was to develop program using Convolution Neural network to extract required information such as name, father’s name, date of birth and PAN number in text form from the legal document called PAN Card. In this project Darknet framework as well as Vott were used
 
 ## Set Up
 1) Clone Darknet
  First of all you need to install Darknet for train our OCR
  to clone darknet use below command:
  => git clone https://github.com/kriyeng/darknet/
 
 2) Set up Darknet Folder
  After cloning Darknet you need to insert pan.py, config.py and utils foldern in darknet folder.
 
 3) Set up for GPU
  Tarining data set using darknet is very time consuming to boost up training you need to use GPU, to set up GPU for Training you need to change setting in makefile
  change below things:
   GPU = 1
   OPENCV = 1
   CUDNN = 1
  then run below command
  make compile.log
  
 
 4) Change Darknet file mode
  For Training You need change access mode of Darknet below command
  => chmod +x darknet
 
 5) Data Annotation :
  Now you need to annoted your data set using Visual object tagging Tool such as Vott. remember you need to generate yolo file and add the output of Visual object tagging Tool to data folder contained in darknet folder.
  
 5) Train Darknet :
  Now you are ready to train your dataset using darknet. use following command to start train you data set:
  => ./darknet detector train data/obj.data yolo-obj.cfg backup/darknet53.conv.74 -dont_show
  
 6) Generate input and results folder:
  After successfully training dataset you need to create two folders called input and results.
  In input folder you need to add images which you want data from that.
  then you need to run command to extrct informations from these images as follows:
  => ./darknet detector test data/obj.data yolo-obj.cfg backup/yolo-obj_18000.weights -thresh 0.25 -ext_output -dont_show
  
  Now, Tadaaa you can see extracted information in output folder.
  
  references: https://medium.com/saarthi-ai/how-to-build-your-own-ocr-a5bb91b622ba
    
