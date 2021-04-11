# PAN-Card-OCR
## Introcution
PAN Card is one type of legal document that is allocated by the income tax department of India to all the taxpayers .

The idea of this project was to develop program using Convolution Neural network to extract required information such as name, father’s name, date of birth and PAN number in text form from the legal document called PAN Card. In this project Darknet framework as well as Vott were used
 
 ## Set Up
 1) Clone Darknet
  First of all you need to install Darknet for train our OCR to clone darknet use below command:<br />
  => git clone https://github.com/kriyeng/darknet/<br /><br />
 
 2) Set up Darknet Folder<br />
  After cloning Darknet you need to insert pan.py, config.py and utils folder in darknet folder.<br /><br />
 
 3) Set up for GPU<br />
  Training a data set using darknet is very time consuming, to boost up training you need to use GPU, to set up GPU for Training you need to change setting in makefile
  change below things:<br />
   GPU = 1<br />
   OPENCV = 1<br />
   CUDNN = 1<br />
  then run below command<br />
  make compile.log<br /><br />
  
 
 4) Change Darknet file mode<br />
  For Training You need change access mode of Darknet using below command<br />
  => chmod +x darknet<br /><br />
 
 5) Data Annotation :<br />
  Now you need to annoted your data set using Visual object tagging Tool such as Vott. remember you need to generate yolo file and add the output of Visual object tagging Tool to data folder contained in darknet folder.<br /><br />
  
 5) Train Darknet :<br />
  Now you are ready to train your dataset using darknet. use following command to start train you data set:<br />
  => ./darknet detector train data/obj.data yolo-obj.cfg backup/darknet53.conv.74 -dont_show<br /><br />
  
 6) Generate input and results folder:<br />
  -> After successfully training dataset you need to create two folders called input and results.<br />
  -> In input folder you need to add images which you want data from that.<br />
  -> then you need to run command to extrct informations from these images as follows:<br />
  => python pan.py -d -t<br /><br />
  
  Now, Tadaaa you can see extracted information in results folder.<br />
  
  references: https://medium.com/saarthi-ai/how-to-build-your-own-ocr-a5bb91b622ba
    
