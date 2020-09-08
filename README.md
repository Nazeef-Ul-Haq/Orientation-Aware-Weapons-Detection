# Orientation Aware Weapons Detection in Visual Data
Automatic detection of weapons is significant for improving security and well-being of individuals, nonetheless, it is a difficult task due to large variety of size, shape and appearance of weapons. View point variations and occlusion also are reasons which makes this task more difficult. Further, the current object detection algorithms process rectangular areas, however a slender and long rifle may really cover just a little portion of area and the rest may contain unessential details. To overcome these problem we propose Orientation Aware Weapons Detection algorithm which provides oriented bounding box and improved detection performance of weapons. The proposed model provides orientation not only using angle as classification problem by dividing angle into eight classes but also angle as regression problem.
![OAWD Architecture](https://github.com/Nazeef-Ul-Haq/Orientation-Aware-Weapons-Detection/architecture.jpg)
# Instructions: 
This code is modified by using Faster RCNN implementation available in Keras. We provide orientation in two ways; one is using angle classification and other is using angle regression.
Kindly see the system setup details of Faster RCNN [Here](https://github.com/kbardool/keras-frcnn ). This will help in running our model. 
We provide necessary files to run the test only using our weights. Weights of model can be downloaded from [Here](https://drive.google.com/file/d/12wVZp-MK5C6rCeWogStyWamCtu-vaTQw/view?usp=sharing).
```python
python test_frcnn.py -p Test
```
# Results:
![Results](https://github.com/Nazeef-Ul-Haq/Orientation-Aware-Weapons-Detection/results.jpg)

# Dataset:
Dataset is available upon request and by filling this google form [Google Form Link](https://docs.google.com/forms/d/e/1FAIpQLSeI_jARiM9Sgjs_dgbfEMHsu_VBuPa_RYZgrdfM8vTL9MnNJQ/viewform?vc=0&c=0&w=1&flr=0&gxids=7757)

