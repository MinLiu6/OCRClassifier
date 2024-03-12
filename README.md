# OCRClassifier
**OCRClassifier: a method combining control charts and machine learning for accurately detecting open states of chromatin**
 In this study,  we present OCRClassifier, a novel framework that combines control charts and machine learning to address the impact of high-proportion noisy labels in the training set and classify the chromatin open states into three classes accurately.
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/985fbe0e258243b9a76782c553bd1a36.jpeg)Flow Chart
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/35162f65e4354401bb7ddfc866e67682.png )Structure of Model A and Model B



@[TOC](Content)
# Project Organization

```
├── LICENSE                     		    <- Non-commercial license.
│     
├── README.md                    		    <- The top-level README for users using DeepLION2.
│ 
├── Codes                        		    <- Python scripts of OCRClassifier. See README for their usages.
│   ├── calWPS_Depth_Endsignal.py 	 		<- Calcute WPS, depth, Uend and Dend signals.
│   ├── NDR.py   
│   ├── Peak.py  
│   ├── predict_OCRClassifier.py 	 	    <- Making predictions
│   ├── getData.py                    	    <- Data preprocessing
│   ├── OCRClassifier_main.py   	        <- Training models.
│   ├── mrcd_HotellingT2.R          	    <- Filtering training datas.
│   ├──MEWMA_ST.R							<-Detecting three classes.
│   ├──pc_calute.py							<-Calcute the probability that a sample belongs to an OCR
│   └── utils.py       		     	        <- Containing the commonly used function components.
│ 
├── Data                         		    <- Data used in OCRClassifier
│   ├── filterByHT
│   │ 
│   ├── gene
│   │   ├── ATAC
│   │   ├── DNase
│   │     
│   └──input
│   │   ├── HT_input						<- Data used in Hotelling T^2 control chart
│   │   ├── ST_input						<- Data used in sensitized T^2 control chart
│   │   ├── testData
│   │   └── trainData
├── Figures                    		        <- Figures used in README.
│   ├── OCRClassifier_workflow.png
│   └── ConvLSTM.png
│  
├── Models                   		        <- Pre-trained OCRCLassifier models for users making predictions directly.        
```
# Usage
## Python package versions
OCRClassifier works perfectly in the following versions of the Python and R packages:

```
Python          3.6.9
matplotlib      3.3.4
numpy           1.19.5
pandas			1.1.5
tensorflow		2.5.0
pysam			0.21.0
R				4.2.1
```
Users can use the Python script `./Codes/predict_OCRClassifier.py` using the pre-trained models we provided in `./Models/OCRClassifier_model_a.h5` and `./Models/OCRClassifier_model_b.h5` to make classification and evaluate the performance.
# Contacts
OCRClassifier is actively maintained by Min Liu, currently a student at Xi'an Jiaotong University in the research group of Prof. Jiayin Wang.
If you have any questions, please contact us by e-mail: lm6080504@stu.xjtu.edu.cn.
