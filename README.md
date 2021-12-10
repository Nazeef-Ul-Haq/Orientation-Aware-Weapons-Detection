# Orientation Aware Weapons Detection in Visual Data
Automatic detection of weapons is significant for improving security and well-being of individuals, nonetheless, it is a difficult task due to large variety of size, shape and appearance of weapons. View point variations and occlusion also are reasons which makes this task more difficult. Further, the current object detection algorithms process rectangular areas, however a slender and long rifle may really cover just a little portion of area and the rest may contain unessential details. To overcome these problems we propose Orientation Aware Weapons Detection algorithm which provides oriented bounding box and improved detection performance of weapons. The proposed model provides orientation not only using angle as classification problem by dividing angle into eight classes but also angle as regression problem.

![OAWD Architecture](https://github.com/Nazeef-Ul-Haq/Orientation-Aware-Weapons-Detection/blob/master/architecture.jpg)

# Instructions: 
This code is modified by using Faster RCNN implementation available in Keras. We provide orientation in two ways; one is using angle classification and other is using angle regression.
Kindly see the system setup details of Faster RCNN [Here](https://github.com/kbardool/keras-frcnn ). This will help in running our model. 
We provide necessary files to run the test only using our weights. Weights of model can be downloaded from [Here](https://drive.google.com/file/d/12wVZp-MK5C6rCeWogStyWamCtu-vaTQw/view?usp=sharing) and put weights into project directory. After downloading weights go to project directory and run this command given below. 
```
python test_frcnn.py -p Test
```
# Results:
![Results](https://github.com/Nazeef-Ul-Haq/Orientation-Aware-Weapons-Detection/blob/master/results.jpg)

# Dataset:
Dataset is available upon request and by filling this google form [Google Form Link](https://docs.google.com/forms/d/e/1FAIpQLSeI_jARiM9Sgjs_dgbfEMHsu_VBuPa_RYZgrdfM8vTL9MnNJQ/viewform?vc=0&c=0&w=1&flr=0&gxids=7757)
# Citation: 
Please cite our papers if you use our dataset in your work. 

1. N.U Haq, M.M. Fraz , T.S Hashmi, M. Shahzad, “Orientation Aware Weapons Detection In Visual Data : A Benchmark Dataset“, 2021, eprint arXiv https://arxiv.org/abs/2112.02221

2. Haq, N. U., Hashmi, T. S. S., Fraz, M. M., & Shahzad, M. (2021, May). Rotation Aware Object Detection Model with Applications to Weapons Spotting in Surveillance Videos. In 2021 International Conference on Digital Futures and Transformative Technologies (ICoDT2) (pp. 1-6). IEEE.

3. Hashmi, T. S. S., Haq, N. U., Fraz, M. M., & Shahzad, M. (2021, May). Application of Deep Learning for Weapons Detection in Surveillance Videos. In 2021 International Conference on Digital Futures and Transformative Technologies (ICoDT2) (pp. 1-6). IEEE.
 
