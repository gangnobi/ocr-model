## Dataset source
[NIST Special Database 19](https://www.nist.gov/srd/nist-special-database-19) -  NIST Handprinted Forms and Characters Database
## How we train?
we used [retrain script](https://github.com/tensorflow/hub/tree/master/examples/image_retraining) of tensorflow to tain with [mobilenet_v2_1.0_224](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet) by this command
```sh
python3 retrain.py --image_dir ~/Documents/NIST_Special_Database_19/tensorflow_form/ --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/1
```
### Train result
Train accuracy = 56.0%
Cross entropy = 1.487925
Validation accuracy = 60.0% (N=100)
