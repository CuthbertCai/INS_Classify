# ** INS_CLassify **

这是一个在Tensorflow上建立的CNN图片分类模型，训练和测试的图片均
来自于对Instagram爬虫获得的图片．为了探究CNN深度以及batch norm-
alization对于性能的影响，分别设计了deep model with batch norma-
lization, simple model with batch normalizaion, simple model w-
thout batch normalization三个模型．

- ins_image_input.py用于将JPG格式的图片转化为tfrecords文件，
为训练和测试提供图片．
- ins_train.py用于对模型进行训练，并记录checkpoint
- ins_eval.py用于对模型进行测试，记录测试时的准确率
- ins_model.py,ins_small_model.py,ins_small_model_bn
分别设计了三种CNN模型用于训练和测试

## 图片来源

本项目的图片是通过在Instagram上[爬虫](https://github.com/CuthbertCai/Instagram)得到的

由于文件大小的限制，本项目制作的tfrecords无法上传至github．可以
通过ins _ image _ input.py自己制作数据集，也可以从网盘下载本项目
的[train.tfrecords](https://mega.nz/#!ErAwjZ7b!eDsQGf0WmmagZnaTYrmqu3T9_r8gOogrXHD0hodhxyk)和[eval.tfrecords](https://mega.nz/#!5jJSwAJY!sgb2zg1tVEpbcrFUCyZ3zvFfe4a_fROApg7HVZdiEWs).

## 参考资料

程序中的函数可以在[Tensorflow](https://www.tensorflow.org/)官网
查找，将图像进行转化和读取可以参考[convert_to_records.py](https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/how_tos/reading_data/convert_to_records.py)
以及[fully_connected_reader.py](https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py).
Tensorflow还有许多详细的[例程](https://github.com/tensorflow/models)可供参考
