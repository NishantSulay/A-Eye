# A-Eye
Object classification on RaspberryPi3 using TensorFlow and Google Cloud Vision 

## Installation guide 
1. Install TensorFlow for RP3 developed by Sam Abrahams(https://github.com/samjabrahams/tensorflow-on-raspberry-pi)
2. Install pico2wave for text-to-speech output to user (http://rpihome.blogspot.ca/2015/02/installing-pico-tts.html)
3. Google Cloud setup found here :https://www.dexterindustries.com/howto/use-google-cloud-vision-on-the-raspberry-pi/


## Training
* Training steps can be varied for higher accuracy however higher accuracy during training can lead to overfitting. Accuracy of 88-93% preferred during training.

* Example below with %userprofile% as destination containing tf_files directory.

```
python tensorflow\tensorflow\examples\image_retraining\retrain.py --bottleneck_dir=%userprofile%\tf_files\bottlenecks  --how_many_training_steps 500 --model_dir=%userprofile%\tf_files\inception --output_graph=%userprofile%\tf_files\retrained_graph.pb --output_labels=%userprofile%\tf_files\retrained_labels.txt --image_dir=%userprofile%\tf_files\dataraining 
```
