import argparse
import sys
def parse_args():
	parser = argparse.ArgumentParser(description=__doc__, \
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('solver_file', help='solver.prototxt')
	parser.add_argument('train_val_file', help='train_val.prototxt')
	args = parser.parse_args()
	return args


def generate_data_layer():
	data_layer = '''name: "ResNet-Imagenet"
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  transform_param {
    crop_size: 299
    mean_value: 104
    mean_value: 117
    mean_value: 123
    mirror: true
  }
  data_param {
    source: "data/ilsvrc12/ilsvrc12_train_lmdb"
    batch_size: 5
    backend: LMDB
  }
}
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TEST
  }
  transform_param {
    crop_size: 299
    mean_value: 104
    mean_value: 117
    mean_value: 123
  }
  data_param {
    source: "data/ilsvrc12/ilsvrc12_test_lmdb"
    batch_size: 5
    backend: LMDB
  }
}
'''
	return data_layer


def generate_conv_bn_scale_act_layer(conv_name, bottom, top, num_output, kernel_size, pad, stride):
	bn_name = conv_name+"_bn"
	scale_name = conv_name+"_scale"
	relu_name = conv_name+"_relu"
	layers = '''layer {
	name: "%s"
	type: "Convolution"
	bottom: "%s"
	top: "%s"
	param {
		lr_mult: 1
		decay_mult: 1
	}
	param {
		lr_mult: 2
		decay_mult: 0
	}
	convolution_param {
		num_output: %d
		kernel_size: %d
		pad: %d
		stride: %d
		weight_filler {
			type: "xavier"
			std: 0.01
		}
		bias_filler {
			type: "constant"
			value: 0.2
		}
	}
}
layer {
	name: "%s"
	type: "BatchNorm"
	bottom: "%s"
	top: "%s"
	batch_norm_param {
		use_global_stats: false
	}
}
layer {
	name: "%s"
	type: "Scale"
	bottom: "%s"
	top: "%s"
	scale_param {
    	bias_term: true
  	}
}
layer {
	name: "%s"
	type: "ReLU"
	bottom: "%s"
	top: "%s"
}
'''%(conv_name, bottom, top, num_output, kernel_size, pad, stride, bn_name, conv_name, conv_name, scale_name, conv_name, conv_name, relu_name, conv_name, conv_name)
	return layers
	

def generate_reconv_bn_scale_act_layer9(conv_name, bottom, top, num_output, kernel_h, kernel_w, pad_h, pad_w, stride):
	bn_name = conv_name+"_bn"
	scale_name = conv_name+"_scale"
	relu_name = conv_name+"_relu"
	layers = '''layer {
	name: "%s"
	type: "Convolution"
	bottom: "%s"
	top: "%s"
	param {
		lr_mult: 1
		decay_mult: 1
	}
	param {
		lr_mult: 2
		decay_mult: 0
	}
	convolution_param {
		num_output: %d
		kernel_h: %d
		kernel_w: %d
		pad_h: %d
		pad_w: %d
		stride: %d
		weight_filler {
			type: "xavier"
			std: 0.01
		}
		bias_filler {
			type: "constant"
			value: 0.2
		}
	}
}
layer {
	name: "%s"
	type: "BatchNorm"
	bottom: "%s"
	top: "%s"
	batch_norm_param {
		use_global_stats: false
	}
}
layer {
	name: "%s"
	type: "Scale"
	bottom: "%s"
	top: "%s"
	scale_param {
    	bias_term: true
  	}
}
layer {
	name: "%s"
	type: "ReLU"
	bottom: "%s"
	top: "%s"
}
'''%(conv_name, bottom, top, num_output, kernel_h, kernel_w, pad_h, pad_w, stride, bn_name, conv_name, conv_name, scale_name, conv_name, conv_name, relu_name, conv_name, conv_name)
	return layers


def generate_conv_bn_scale_layer(conv_name, bottom, top, num_output, kernel_size, pad, stride):
	bn_name = conv_name+"_bn"
	scale_name = conv_name+"_scale"
	layers = '''layer {
	name: "%s"
	type: "Convolution"
	bottom: "%s"
	top: "%s"
	param {
		lr_mult: 1
		decay_mult: 1
	}
	param {
		lr_mult: 2
		decay_mult: 0
	}
	convolution_param {
		num_output: %d
		kernel_size: %d
		pad: %d
		stride: %d
		weight_filler {
			type: "xavier"
			std: 0.01
		}
		bias_filler {
			type: "constant"
			value: 0.2
		}
	}
}
layer {
	name: "%s"
	type: "BatchNorm"
	bottom: "%s"
	top: "%s"
	batch_norm_param {
		use_global_stats: false
	}
}
layer {
	name: "%s"
	type: "Scale"
	bottom: "%s"
	top: "%s"
	scale_param {
    	bias_term: true
  	}
}
'''%(conv_name, bottom, top, num_output, kernel_size, pad, stride, bn_name, conv_name, conv_name, scale_name, conv_name, conv_name)
	return layers


def generate_eltwise_layer(eltwise_name, bottom1, bottom2):
	eltwise_layer = '''layer {
	name: "%s"
	bottom: "%s"
	bottom: "%s"
	top: "%s"
	type: "Eltwise"
	eltwise_param {
		operation: SUM
	}
}
'''%(eltwise_name, bottom1, bottom2, eltwise_name)
	return eltwise_layer


def generate_concat_layer2(name, bottom1, bottom2):
	concat_layer = '''layer {
	name: "%s"
	type: "Concat"
	bottom: "%s"
	bottom: "%s"
	top: "%s"
}
'''%(name, bottom1, bottom2, name)
	return concat_layer


def generate_concat_layer3(name, bottom1, bottom2, bottom3):
	concat_layer = '''layer {
	name: "%s"
	type: "Concat"
	bottom: "%s"
	bottom: "%s"
	bottom: "%s"
	top: "%s"
}
'''%(name, bottom1, bottom2, bottom3, name)
	return concat_layer


def generate_concat_layer4(name, bottom1, bottom2, bottom3, bottom4):
	concat_layer = '''layer {
	name: "%s"
	type: "Concat"
	bottom: "%s"
	bottom: "%s"
	bottom: "%s"
	bottom: "%s"
	top: "%s"
}
'''%(name, bottom1, bottom2, bottom3, bottom4, name)
	return concat_layer


def generate_pooling_layer(name, bottom, top, pool_type, kernel_size, stride):
	pooling_layer = '''layer {
	name: "%s"
	type: "Pooling"
	bottom: "%s"
	top: "%s"
	pooling_param {
		pool: %s
		kernel_size: %d
		stride: %d
	}
}
'''%(name, bottom, top, pool_type, kernel_size, stride)
	return pooling_layer


def generate_pool_dropout_layer(pool_name, bottom, dropout_name):
	layers = '''layer {
	name: "%s"
	type: "Pooling"
	bottom: "%s"
	top: "%s"
	pooling_param {
		pool: AVE
		global_pooling: true
	}
}
layer {
	name: "%s"
	type: "Dropout"
	bottom: "%s"
	top: "%s"
	dropout_param {
		dropout_ratio: 0.2
	}
}
'''%(pool_name, bottom, pool_name, dropout_name, pool_name, dropout_name)
	return layers


def generate_fc_softmax_accuracy_layer(name, bottom, num_output, filler_type="msra"):
	layers = '''layer {
	name: "%s"
	type: "InnerProduct"
	bottom: "%s"
	top: "%s"
	param {
		lr_mult: 1
		decay_mult: 1
	}
	param {
		lr_mult: 2
		decay_mult: 0
	}
	inner_product_param {
		num_output: %d
		weight_filler {
			type: "%s"
			std: 0.01
		}
		bias_filler {
			type: "constant"
			value: 0
		}
	}
}
layer {
	name: "SoftmaxWithLoss1"
	type: "SoftmaxWithLoss"
	bottom: "%s"
	bottom: "label"
	top: "SoftmaxWithLoss1"
}
layer {
	name: "Accuracy1"
	type: "Accuracy"
	bottom: "%s"
	bottom: "label"
	top: "Accuracy1"
	include {
		phase: TEST
	}
}
layer {
	name: "Accuracy5"
	type: "Accuracy"
	bottom: "%s"
	bottom: "label"
	top: "Accuracy5"
	include {
		phase: TEST
	}
	accuracy_param {
		top_k: 5
	}
}
'''%(name, bottom, name, num_output, filler_type, name, name, name)
	return layers


def generate_stem():
	stem = ""
	stem += generate_data_layer()
	stem += generate_conv_bn_scale_act_layer("conv1_3x3", "data", "conv1_3x3", 32, 3, 0, 2)
	stem += generate_conv_bn_scale_act_layer("conv2_3x3", "conv1_3x3", "conv2_3x3", 32, 3, 0, 1)
	stem += generate_conv_bn_scale_act_layer("conv3_3x3", "conv2_3x3", "conv3_3x3", 64, 3, 1, 1)
	stem += generate_pooling_layer("pool1_3x3", "conv3_3x3", "pool1_3x3", "MAX", 3, 2)
	stem += generate_conv_bn_scale_act_layer("conv4_3x3", "conv3_3x3", "conv4_3x3", 96, 3, 0, 2)
	stem += generate_concat_layer2("stem_concat_1", "pool1_3x3", "conv4_3x3")
	stem += generate_conv_bn_scale_act_layer("inception_reduction_conv1_1x1", "stem_concat_1", "inception_reduction_conv1_1x1", 64, 1, 0, 1)
	stem += generate_conv_bn_scale_act_layer("inception_reduction_conv2_3x3", "inception_reduction_conv1_1x1", "inception_reduction_conv2_3x3", 96, 3, 0, 1)
	stem += generate_conv_bn_scale_act_layer("inception_reduction_conv3_1x1", "stem_concat_1", "inception_reduction_conv3_1x1", 64, 1, 0, 1)
	stem += generate_reconv_bn_scale_act_layer9("inception_reduction_conv4_7x1", "inception_reduction_conv3_1x1", "inception_reduction_conv4_7x1", 64, 7, 1, 3, 0, 1)
	stem += generate_reconv_bn_scale_act_layer9("inception_reduction_conv5_1x7", "inception_reduction_conv4_7x1", "inception_reduction_conv5_1x7", 64, 1, 7, 0, 3, 1)
	stem += generate_conv_bn_scale_act_layer("inception_reduction_conv6_3x3", "inception_reduction_conv5_1x7", "inception_reduction_conv6_3x3", 96, 3, 0, 1)
	stem += generate_concat_layer2("stem_concat_2", "inception_reduction_conv2_3x3", "inception_reduction_conv6_3x3")
	stem += generate_conv_bn_scale_act_layer("conv5_3x3", "stem_concat_2", "conv5_3x3", 192, 3, 0, 2)
	stem += generate_pooling_layer("pool2_3x3", "stem_concat_2", "pool2_3x3", "MAX", 3, 2)
	stem += generate_concat_layer2("stem_concat_3", "conv5_3x3", "pool2_3x3")
	last_layer = "stem_concat_3"
	return stem, last_layer


def generate_inception_resnet_a(index, last_layer):
	net = ""
	net += generate_conv_bn_scale_act_layer("inception_"+str(index)+"_conv1_1x1", last_layer, "inception_"+str(index)+"_conv1_1x1", 32, 1, 0, 1)
	net += generate_conv_bn_scale_act_layer("inception_"+str(index)+"_conv2_1_1x1", last_layer, "inception_"+str(index)+"_conv2_1_1x1", 32, 1, 0, 1)
	net += generate_conv_bn_scale_act_layer("inception_"+str(index)+"_conv2_2_1x1", "inception_"+str(index)+"_conv2_1_1x1", "inception_"+str(index)+"_conv2_2_1x1", 32, 3, 1, 1)
	net += generate_conv_bn_scale_act_layer("inception_"+str(index)+"_conv3_1_1x1", last_layer, "inception_"+str(index)+"_conv3_1_1x1", 32, 1, 0, 1)
	net += generate_conv_bn_scale_act_layer("inception_"+str(index)+"_conv3_2_1x1", "inception_"+str(index)+"_conv3_1_1x1", "inception_"+str(index)+"_conv3_2_1x1", 48, 3, 1, 1)
	net += generate_conv_bn_scale_act_layer("inception_"+str(index)+"_conv3_3_1x1", "inception_"+str(index)+"_conv3_2_1x1", "inception_"+str(index)+"_conv3_3_1x1", 64, 3, 1, 1)
	net += generate_concat_layer3("inception_"+str(index)+"_concat", "inception_"+str(index)+"_conv1_1x1", "inception_"+str(index)+"_conv2_1_1x1", "inception_"+str(index)+"_conv3_1_1x1")
	net += generate_conv_bn_scale_layer("inception_"+str(index)+"_conv_1x1", "inception_"+str(index)+"_concat", "inception_"+str(index)+"_conv_1x1", 384, 1, 0, 1)
	net += generate_eltwise_layer("inception_"+str(index)+"_eltwise", last_layer, "inception_"+str(index)+"_conv_1x1")
	last_layer = "inception_"+str(index)+"_eltwise"
	return net, last_layer


def generate_reduction_a(last_layer):
	net = ""
	net += generate_pooling_layer("reduction_a_pool", last_layer, "reduction_a_pool", "MAX", 3, 2)
	net += generate_conv_bn_scale_act_layer("reduction_a_conv1_3x3", last_layer, "reduction_a_conv1_3x3", 384, 3, 0, 2)
	net += generate_conv_bn_scale_act_layer("reduction_a_conv2_1_1x1", last_layer, "reduction_a_conv2_1_1x1", 256, 1, 0, 1)
	net += generate_conv_bn_scale_act_layer("reduction_a_conv2_2_3x3", "reduction_a_conv2_1_1x1", "reduction_a_conv2_2_3x3", 256, 3, 1, 1)
	net += generate_conv_bn_scale_act_layer("reduction_a_conv2_3_3x3", "reduction_a_conv2_2_3x3", "reduction_a_conv2_3_3x3", 384, 3, 0, 2)
	net += generate_concat_layer3("reduction_a_concat", "reduction_a_pool", "reduction_a_conv1_3x3", "reduction_a_conv2_3_3x3")
	last_layer = "reduction_a_concat"
	return net, last_layer


def generate_inception_resnet_b(index, last_layer):
	net = ""
	net += generate_conv_bn_scale_act_layer("inception_b_"+str(index)+"_conv1_1x1", last_layer, "inception_b_"+str(index)+"_conv1_1x1", 192, 1, 0, 1)
	net += generate_conv_bn_scale_act_layer("inception_b_"+str(index)+"_conv2_1_1x1", last_layer, "inception_b_"+str(index)+"_conv2_1_1x1", 128, 1, 0, 1)
	net += generate_reconv_bn_scale_act_layer9("inception_b_"+str(index)+"_conv2_2_1x7", "inception_b_"+str(index)+"_conv2_1_1x1", "inception_b_"+str(index)+"_conv2_2_1x7", 160, 1, 7, 0, 3, 1)
	net += generate_reconv_bn_scale_act_layer9("inception_b_"+str(index)+"_conv2_3_7x1", "inception_b_"+str(index)+"_conv2_2_1x7", "inception_b_"+str(index)+"_conv2_3_7x1", 192, 7, 1, 3, 0, 1)
	net += generate_concat_layer2("inception_b_"+str(index)+"_concat", "inception_b_"+str(index)+"_conv1_1x1", "inception_b_"+str(index)+"_conv2_3_7x1")
	net += generate_conv_bn_scale_layer("inception_b_"+str(index)+"_conv3_1x1", "inception_b_"+str(index)+"_concat", "inception_b_"+str(index)+"_conv3_1x1", 1152, 1, 0, 1)
	net += generate_eltwise_layer("inception_b_"+str(index)+"_eltwise", last_layer, "inception_b_"+str(index)+"_conv3_1x1")
	last_layer = "inception_b_"+str(index)+"_eltwise"
	return net, last_layer


def generate_reduction_b(last_layer):
	net = ""
	net += generate_pooling_layer("reduction_b_pool", last_layer, "reduction_b_pool", "MAX", 3, 2)
	net += generate_conv_bn_scale_act_layer("reduction_b_conv1_1_1x1", last_layer, "reduction_b_conv1_1_1x1", 256, 1, 0, 1)
	net += generate_conv_bn_scale_act_layer("reduction_b_conv1_2_3x3", "reduction_b_conv1_1_1x1", "reduction_b_conv1_2_3x3", 384, 3, 0, 2)
	net += generate_conv_bn_scale_act_layer("reduction_b_conv2_1_1x1", last_layer, "reduction_b_conv2_1_1x1", 256, 1, 0, 1)
	net += generate_conv_bn_scale_act_layer("reduction_b_conv2_2_3x3", "reduction_b_conv2_1_1x1", "reduction_b_conv2_2_3x3", 256, 3, 0, 2)
	net += generate_conv_bn_scale_act_layer("reduction_b_conv3_1_1x1", last_layer, "reduction_b_conv3_1_1x1", 256, 1, 0, 1)
	net += generate_conv_bn_scale_act_layer("reduction_b_conv3_2_3x3", "reduction_b_conv3_1_1x1", "reduction_b_conv3_2_3x3", 256, 3, 1, 1)
	net += generate_conv_bn_scale_act_layer("reduction_b_conv3_3_3x3", "reduction_b_conv3_2_3x3", "reduction_b_conv3_3_3x3", 256, 3, 0, 2)
	net += generate_concat_layer4("reduction_b_concat", "reduction_b_pool", "reduction_b_conv1_2_3x3", "reduction_b_conv2_2_3x3", "reduction_b_conv3_3_3x3")
	last_layer = "reduction_b_concat"
	return net, last_layer


def generate_inception_resnet_c(index, last_layer):
	net = ""
	net += generate_conv_bn_scale_act_layer("inception_c_"+str(index)+"_conv1_1x1", last_layer, "inception_c_"+str(index)+"_conv1_1x1", 192, 1, 0, 1)
	net += generate_conv_bn_scale_act_layer("inception_c_"+str(index)+"_conv2_1_1x1", last_layer, "inception_c_"+str(index)+"_conv2_1_1x1", 192, 1, 0, 1)
	net += generate_reconv_bn_scale_act_layer9("inception_c_"+str(index)+"_conv2_2_1x3", "inception_c_"+str(index)+"_conv2_1_1x1", "inception_c_"+str(index)+"_conv2_2_1x3", 224, 1, 3, 0, 1, 1)
	net += generate_reconv_bn_scale_act_layer9("inception_c_"+str(index)+"_conv2_3_3x1", "inception_c_"+str(index)+"_conv2_2_1x3", "inception_c_"+str(index)+"_conv2_3_3x1", 256, 3, 1, 1, 0, 1)
	net += generate_concat_layer2("inception_c_"+str(index)+"_concat", "inception_c_"+str(index)+"_conv1_1x1", "inception_c_"+str(index)+"_conv2_3_3x1")
	net += generate_conv_bn_scale_layer("inception_c_"+str(index)+"_conv3_1x1", "inception_c_"+str(index)+"_concat", "inception_c_"+str(index)+"_conv3_1x1", 2048, 1, 0, 1)
	net += generate_eltwise_layer("inception_c_"+str(index)+"_eltwise", last_layer, "inception_c_"+str(index)+"_conv3_1x1")
	last_layer = "inception_c_"+str(index)+"_eltwise"
	return net, last_layer


def generate_train_val():
	train_val_file = ""
	stem, last_layer = generate_stem()
	train_val_file += stem
	for i in range(1, 6):
		words, last_layer = generate_inception_resnet_a(i, last_layer)
		train_val_file += words
	words, last_layer = generate_reduction_a(last_layer)
	train_val_file += words
	for i in range(1, 11):
		words, last_layer = generate_inception_resnet_b(i, last_layer)
		train_val_file += words
	words, last_layer = generate_reduction_b(last_layer)
	train_val_file += words
	for i in range(1, 6):
		words, last_layer = generate_inception_resnet_c(i, last_layer)
		train_val_file += words
	train_val_file += generate_pool_dropout_layer("pool1", last_layer, "dropout")
	train_val_file += generate_fc_softmax_accuracy_layer("fc1000", "dropout", 1000)
	return train_val_file


def generate_solver(train_val_file):
	solver_file = '''net: "%s"
test_iter: 1000
test_interval: 1000
test_initialization: false

base_lr: 0.001
lr_policy: "multistep"
gamma: 0.1
stepvalue: 60000
stepvalue: 120000
max_iter: 600000
weight_decay: 0.0001
momentum: 0.9

display: 100
snapshot: 10000
snapshot_prefix: "resnet-imagenet"
solver_mode: GPU'''%(train_val_file)
	return solver_file


if __name__ == '__main__':
	args = parse_args()
	train_val_file = args.train_val_file
	solver_file = generate_solver(train_val_file)
	network_file = generate_train_val()
	fw = open(args.solver_file, 'w+')
	fw.write(solver_file)
	fw.close()
	fw = open(args.train_val_file, 'w+')
	fw.write(network_file)
	fw.close()