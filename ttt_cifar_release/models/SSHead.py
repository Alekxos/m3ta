from torch import nn
import math
import copy
import functional_layers as L
import torch
torch.autograd.set_detect_anomaly(True)

class ViewFlatten(nn.Module):
	def __init__(self):
		super(ViewFlatten, self).__init__()

	def forward(self, x):
		return x.view(x.size(0), -1)

class ExtractorHead(nn.Module):
	def __init__(self, ext, head):
		super(ExtractorHead, self).__init__()
		self.ext = ext
		self.head = head

	def forward(self, input, weights=None, cuda=True, double=False):
		input = input.double()
		if cuda:
			input.cuda()
		if weights == None:
			output = self.head(self.ext(input))
		else:
			# for layer in self.layers:
			# 	print(f"layer: {layer}")
			# print(f"weights: {weights.keys()}")
			output = input
			# net.conv1
			output = L.conv2d(output, weights[f'ext.0.weight'], cuda=cuda)
			# layers 1, 2 (ext)
			for ext_seq_id in range(1, 3):
				identity = output
				for ext_block_id in range(4):
					for sub_block_id in range(1, 3):
						stride = 2 if (ext_seq_id == 2 and ext_block_id == 0 and sub_block_id == 1) else 1
						# error on next line?
						output = L.batch_norm(output, weights[f'ext.{ext_seq_id}.{ext_block_id}.bn{sub_block_id}.weight'],
											  bias=weights[f'ext.{ext_seq_id}.{ext_block_id}.bn{sub_block_id}.bias'],
											  momentum=0.1, cuda=cuda)
						output = L.relu(output)
						output = L.conv2d(output, weights[f'ext.{ext_seq_id}.{ext_block_id}.conv{sub_block_id}.weight'], stride=stride, cuda=cuda)

				# downsample
				if ext_seq_id == 2:
					identity = L.avg_pool(identity, 2)
					identity = torch.cat([identity] + [identity.mul(0)], 1)
				output += identity
			# layer 3 (head)
			head_seq_id = 0
			identity = output

			for head_block_id in range(4):
				for sub_block_id in range(1, 3):
					stride = 2 if (head_block_id == 0 and sub_block_id == 1) else 1
					output = L.batch_norm(output,
											weights[f'head.{head_seq_id}.{head_block_id}.bn{sub_block_id}.weight'],
											bias=weights[f'head.{head_seq_id}.{head_block_id}.bn{sub_block_id}.bias'],
											momentum=0.1, cuda=cuda)
					output = L.relu(output)
					output = L.conv2d(output, weights[f'head.{head_seq_id}.{head_block_id}.conv{sub_block_id}.weight'],
										stride=stride, cuda=cuda)
			# downsample
			identity = L.avg_pool(identity, 2)
			identity = torch.cat([identity] + [identity.mul(0)], 1)
			output += identity

			output = L.batch_norm(output, weights['head.1.weight'], bias=weights['head.1.bias'], momentum=0.1, cuda=cuda)
			output = L.relu(output)
			output = L.avg_pool(output, 8)

			output = output.view(output.size(0), -1)
			output = L.linear(output, weights['head.5.weight'], bias=weights['head.5.bias'], cuda=cuda)

		return output

def extractor_from_layer3(net):
	layers = [net.conv1, net.layer1, net.layer2, net.layer3, net.bn, net.relu, net.avgpool, ViewFlatten()]
	return nn.Sequential(*layers)

def extractor_from_layer2(net):
	layers = [net.conv1, net.layer1, net.layer2]
	return nn.Sequential(*layers)

def head_on_layer2(net, width, classes):
	head = copy.deepcopy([net.layer3, net.bn, net.relu, net.avgpool])
	head.append(ViewFlatten())
	head.append(nn.Linear(64 * width, classes))
	return nn.Sequential(*head)
