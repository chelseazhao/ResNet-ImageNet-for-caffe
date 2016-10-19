import argparse
import sys
def parse_args():
	parser = argparse.ArgumentParser(description=__doc__, \
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('solver_file', help='solver.prototxt')
	parser.add_argument('train_val_file', help='train_val.prototxt')
	parser.add_argument('--layer_num', nargs=1, help=('the sum of layers, it should be 50, 101, 152 or 200'), \
		default=152)
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
    crop_size: 224
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
    crop_size: 224
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


def generate_conv_layer(name, bottom, top, num_output, kernel_size, pad, stride, filler_type="msra"):
	conv_layer = '''layer {
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
			type: "%s"
		}
		bias_filler {
			type: "constant"
			value: 0
		}
	}
}
'''%(name, bottom, top, num_output, kernel_size, pad, stride, filler_type)
	return conv_layer
	

def generate_BN_layer(name, bottom, top):
	BN_layer = '''layer {
	name: "%s"
	type: "BatchNorm"
	bottom: "%s"
	top: "%s"
	batch_norm_param {
		use_global_stats: false
	}
	param {
		lr_mult: 0
	}
	param {
		lr_mult: 0
	}
	param {
		lr_mult: 0
	}
}
'''%(name, bottom, top)
	return BN_layer


def generate_scale_layer(name, bottom, top):
	scale_layer = '''layer {
	name: "%s"
	type: "Scale"
	bottom: "%s"
	top: "%s"
	scale_param {
    	filler {
      		value: 1
    	}	
    	bias_term: true
    	bias_filler {
      		value: 1
    	}
  	}
}
'''%(name, bottom, top)
	return scale_layer


def generate_eltwise_layer(name, bottom1, bottom2, top, op_type="SUM"):
	eltwise_layer = '''layer {
	name: "%s"
	bottom: "%s"
	bottom: "%s"
	top: "%s"
	type: "Eltwise"
	eltwise_param {
		operation: %s
	}
}
'''%(name, bottom1, bottom2, top, op_type)
	return eltwise_layer


def generate_activation_layer(name, bottom, top, act_type="ReLU"):
	activation_layer = '''layer {
	name: "%s"
	type: "%s"
	bottom: "%s"
	top: "%s"
}
'''%(name, act_type, bottom, top)
	return activation_layer


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


def generate_fc_layer(name, bottom, top, num_output, filler_type="msra"):
	fc_layer = '''layer {
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
'''%(name, bottom, top, num_output, filler_type)
	return fc_layer


def generate_softmaxloss_layer(bottom):
	softmaxloss_layer = '''layer {
	name: "SoftmaxWithLoss1"
	type: "SoftmaxWithLoss"
	bottom: "%s"
	bottom: "label"
	top: "SoftmaxWithLoss1"
}
'''%(bottom)
	return softmaxloss_layer


def generate_accuracy_layer(bottom):
	accuracy_layer = '''layer {
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
'''%(bottom, bottom)
	return accuracy_layer


def generate_train_val(layer_sum):
	train_val_file = ""
	train_val_file += generate_data_layer()
	'''conv1'''
	bottom = 'data'
	train_val_file += generate_conv_layer('conv1', bottom, 'conv1', 64, 7, 3, 2)
	train_val_file += generate_BN_layer('bn_conv1', 'conv1', 'conv1')
	train_val_file += generate_scale_layer('scale_conv1', 'conv1', 'conv1')
	train_val_file += generate_activation_layer('relu_conv1', 'conv1', 'conv1')
	train_val_file += generate_pooling_layer('pool_conv1', 'conv1', 'pool_conv1', 'MAX', 3, 2)
	bottom = 'pool_conv1'

	if layer_sum == 50:
		params = [3, 4, 6, 3]
	elif layer_sum == 101:
		params = [3, 4, 23, 3]
	elif layer_sum == 152:
		params = [3, 8, 36, 3]
	else:
		params = [3, 24, 36, 3]
	'''conv2_x'''
	train_val_file += generate_conv_layer('res2_1_branch1', bottom, 'res2_1_branch1', 256, 1, 0, 1)
	last_out = 'res2_1_branch1'

	this_name = 'res2_1_branch2a'
	train_val_file += generate_conv_layer(this_name, bottom, this_name, 64, 1, 0, 1)
	train_val_file += generate_BN_layer('bn_'+this_name, this_name, this_name)
	train_val_file += generate_scale_layer('scale_'+this_name, this_name, this_name)
	train_val_file += generate_activation_layer('relu_'+this_name, this_name, this_name)

	this_name = 'res2_1_branch2b'
	train_val_file += generate_conv_layer(this_name, 'res2_1_branch2a', this_name, 64, 3, 1, 1)
	train_val_file += generate_BN_layer('bn_'+this_name, this_name, this_name)
	train_val_file += generate_scale_layer('scale_'+this_name, this_name, this_name)
	train_val_file += generate_activation_layer('relu_'+this_name, this_name, this_name)

	this_name = 'res2_1_branch2c'
	train_val_file += generate_conv_layer(this_name, 'res2_1_branch2b', this_name, 256, 1, 0, 1)
	train_val_file += generate_BN_layer('bn_'+this_name, this_name, this_name)
	train_val_file += generate_scale_layer('scale_'+this_name, this_name, this_name)

	train_val_file += generate_eltwise_layer('res2_1', last_out, this_name, 'res2_1')
	last_out = 'res2_1'
	bottom = last_out

	for i in range(2, params[0]+1):
		this_name = 'res2_'+str(i)+'_branch2a_pre'
		train_val_file += generate_BN_layer('bn_'+this_name, bottom, this_name)
		train_val_file += generate_scale_layer('scale_'+this_name, this_name, this_name)
		train_val_file += generate_activation_layer('relu_'+this_name, this_name, this_name)
		bottom = this_name

		this_name = 'res2_'+str(i)+'_branch2a'
		train_val_file += generate_conv_layer(this_name, bottom, this_name, 64, 1, 0, 1)
		train_val_file += generate_BN_layer('bn_'+this_name, this_name, this_name)
		train_val_file += generate_scale_layer('scale_'+this_name, this_name, this_name)
		bottom = this_name

		this_name = 'res2_'+str(i)+'_branch2b'
		train_val_file += generate_conv_layer(this_name, bottom, this_name, 64, 3, 1, 1)
		train_val_file += generate_BN_layer('bn_'+this_name, this_name, this_name)
		train_val_file += generate_scale_layer('scale_'+this_name, this_name, this_name)
		bottom = this_name

		this_name = 'res2_'+str(i)+'_branch2c'
		train_val_file += generate_conv_layer(this_name, bottom, this_name, 256, 1, 0, 1)
		train_val_file += generate_eltwise_layer('res2_'+str(i), last_out, this_name, 'res2_'+str(i))
		last_out = 'res2_'+str(i)
		bottom = last_out
	'''conv3_x'''
	train_val_file += generate_BN_layer('bn_res3_1_branch1_pre', bottom, 'res3_1_branch1_pre')
	train_val_file += generate_scale_layer('scale_res3_1_branch1_pre', 'res3_1_branch1_pre', 'res3_1_branch1_pre')
	train_val_file += generate_activation_layer('relu_res3_1_branch1_pre', 'res3_1_branch1_pre', 'res3_1_branch1_pre')
	train_val_file += generate_conv_layer('res3_1_branch1', 'res3_1_branch1_pre', 'res3_1_branch1', 512, 1, 0, 2)
	
	last_out = 'res3_1_branch1'
	for i in range(1, params[0]+1):
		if i == 1:
			first_stride = 2
		else:
			first_stride = 1
		this_name = 'res3_'+str(i)+'_branch2a_pre'
		train_val_file += generate_BN_layer('bn_'+this_name, bottom, this_name)
		train_val_file += generate_scale_layer('scale_'+this_name, this_name, this_name)
		train_val_file += generate_activation_layer('relu_'+this_name, this_name, this_name)
		bottom = this_name

		this_name = 'res3_'+str(i)+'_branch2a'
		train_val_file += generate_conv_layer(this_name, bottom, this_name, 128, 1, 0, 1)
		train_val_file += generate_BN_layer('bn_'+this_name, this_name, this_name)
		train_val_file += generate_scale_layer('scale_'+this_name, this_name, this_name)
		bottom = this_name

		this_name = 'res3_'+str(i)+'_branch2b'
		train_val_file += generate_conv_layer(this_name, bottom, this_name, 128, 3, 1, first_stride)
		train_val_file += generate_BN_layer('bn_'+this_name, this_name, this_name)
		train_val_file += generate_scale_layer('scale_'+this_name, this_name, this_name)
		bottom = this_name

		this_name = 'res3_'+str(i)+'_branch2c'
		train_val_file += generate_conv_layer(this_name, bottom, this_name, 512, 1, 0, 1)
		train_val_file += generate_eltwise_layer('res3_'+str(i), last_out, this_name, 'res3_'+str(i))
		last_out = 'res3_'+str(i)
		bottom = last_out
	'''conv4_x'''
	train_val_file += generate_BN_layer('bn_res4_1_branch1_pre', bottom, 'res4_1_branch1_pre')
	train_val_file += generate_scale_layer('scale_res4_1_branch1_pre', 'res4_1_branch1_pre', 'res4_1_branch1_pre')
	train_val_file += generate_activation_layer('relu_res4_1_branch1_pre', 'res4_1_branch1_pre', 'res4_1_branch1_pre')
	train_val_file += generate_conv_layer('res4_1_branch1', 'res4_1_branch1_pre', 'res4_1_branch1', 1024, 1, 0, 2)
	
	last_out = 'res4_1_branch1'
	for i in range(1, params[0]+1):
		if i == 1:
			first_stride = 2
		else:
			first_stride = 1
		this_name = 'res4_'+str(i)+'_branch2a_pre'
		train_val_file += generate_BN_layer('bn_'+this_name, bottom, this_name)
		train_val_file += generate_scale_layer('scale_'+this_name, this_name, this_name)
		train_val_file += generate_activation_layer('relu_'+this_name, this_name, this_name)
		bottom = this_name

		this_name = 'res4_'+str(i)+'_branch2a'
		train_val_file += generate_conv_layer(this_name, bottom, this_name, 256, 1, 0, 1)
		train_val_file += generate_BN_layer('bn_'+this_name, this_name, this_name)
		train_val_file += generate_scale_layer('scale_'+this_name, this_name, this_name)
		bottom = this_name

		this_name = 'res4_'+str(i)+'_branch2b'
		train_val_file += generate_conv_layer(this_name, bottom, this_name, 256, 3, 1, first_stride)
		train_val_file += generate_BN_layer('bn_'+this_name, this_name, this_name)
		train_val_file += generate_scale_layer('scale_'+this_name, this_name, this_name)
		bottom = this_name

		this_name = 'res4_'+str(i)+'_branch2c'
		train_val_file += generate_conv_layer(this_name, bottom, this_name, 1024, 1, 0, 1)
		train_val_file += generate_eltwise_layer('res4_'+str(i), last_out, this_name, 'res4_'+str(i))
		last_out = 'res4_'+str(i)
		bottom = last_out
	'''conv5_x'''
	train_val_file += generate_BN_layer('bn_res5_1_branch1_pre', bottom, 'res5_1_branch1_pre')
	train_val_file += generate_scale_layer('scale_res5_1_branch1_pre', 'res5_1_branch1_pre', 'res5_1_branch1_pre')
	train_val_file += generate_activation_layer('relu_res5_1_branch1_pre', 'res5_1_branch1_pre', 'res5_1_branch1_pre')
	train_val_file += generate_conv_layer('res5_1_branch1', 'res5_1_branch1_pre', 'res5_1_branch1', 2048, 1, 0, 2)
	
	last_out = 'res5_1_branch1'
	for i in range(1, params[0]+1):
		if i == 1:
			first_stride = 2
		else:
			first_stride = 1
		this_name = 'res5_'+str(i)+'_branch2a_pre'
		train_val_file += generate_BN_layer('bn_'+this_name, bottom, this_name)
		train_val_file += generate_scale_layer('scale_'+this_name, this_name, this_name)
		train_val_file += generate_activation_layer('relu_'+this_name, this_name, this_name)
		bottom = this_name

		this_name = 'res5_'+str(i)+'_branch2a'
		train_val_file += generate_conv_layer(this_name, bottom, this_name, 512, 1, 0, 1)
		train_val_file += generate_BN_layer('bn_'+this_name, this_name, this_name)
		train_val_file += generate_scale_layer('scale_'+this_name, this_name, this_name)
		bottom = this_name

		this_name = 'res5_'+str(i)+'_branch2b'
		train_val_file += generate_conv_layer(this_name, bottom, this_name, 512, 3, 1, first_stride)
		train_val_file += generate_BN_layer('bn_'+this_name, this_name, this_name)
		train_val_file += generate_scale_layer('scale_'+this_name, this_name, this_name)
		bottom = this_name

		this_name = 'res5_'+str(i)+'_branch2c'
		train_val_file += generate_conv_layer(this_name, bottom, this_name, 2048, 1, 0, 1)
		train_val_file += generate_eltwise_layer('res5_'+str(i), last_out, this_name, 'res5_'+str(i))
		last_out = 'res5_'+str(i)
		bottom = last_out

	this_name = 'res5'
	train_val_file += generate_BN_layer('bn_'+this_name, bottom, this_name)
	train_val_file += generate_scale_layer('scale_'+this_name, this_name, this_name)
	train_val_file += generate_activation_layer('relu_'+this_name, this_name, this_name)
	bottom = this_name

	train_val_file += generate_pooling_layer('pool2', bottom, 'pool2', 'AVE', 7, 1)
	train_val_file += generate_fc_layer('fc1000', 'pool2', 'fc1000', 1000, 'gaussian')
	train_val_file += generate_softmaxloss_layer('fc1000')
	train_val_file += generate_accuracy_layer('fc1000')

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
	layer_sum = int(args.layer_num[0])
	train_val_file = args.train_val_file
	if layer_sum not in [50, 101, 152, 200]:
		print "layer num is not in [50, 101, 152, 200]"
		sys.exit(1)
	solver_file = generate_solver(train_val_file)
	network_file = generate_train_val(layer_sum)
	fw = open(args.solver_file, 'w+')
	fw.write(solver_file)
	fw.close()
	fw = open(args.train_val_file, 'w+')
	fw.write(network_file)
	fw.close()