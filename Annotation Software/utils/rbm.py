import torch
from torch import nn
from torch.autograd import Variable
import random


class RBM_Cell(nn.Module):

	def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias, p_TD):
		super(RBM_Cell, self).__init__()

		self.height, self.width = input_size
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim

		self.kernel_size = kernel_size
		self.padding = kernel_size[0] // 2, kernel_size[1] // 2  # divide exactly
		self.bias = bias

		self.p_TD = p_TD

		self.data_cnn = nn.Conv2d(in_channels=self.input_dim,
		                          out_channels=self.hidden_dim,
		                          kernel_size=self.kernel_size,
		                          padding=self.padding,
		                          bias=self.bias)
		self.ctrl_cnn = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
		                          out_channels=self.hidden_dim,
		                          kernel_size=self.kernel_size,
		                          padding=self.padding,
		                          bias=self.bias)

	def forward(self, input_tensor, cur_state):
		rate = random.random()

		c = cur_state
		data_x = input_tensor
		ctrl_x = input_tensor.detach() if rate < self.p_TD else input_tensor
		ctrl_in = torch.cat((c, ctrl_x), dim=1)

		data_out = torch.tanh(self.data_cnn(data_x))
		ctrl_out = torch.sigmoid(self.ctrl_cnn(ctrl_in))

		return ctrl_out * data_out, ctrl_out

	def init_hidden(self, batch_size):
		return Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width))


class RBM(nn.Module):

	def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers, p_TD,
	             batch_first=False, bias=True, return_all_layers=False):
		super(RBM, self).__init__()

		self._check_kernel_size_consistency(kernel_size)

		# Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
		kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
		hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
		if not len(kernel_size) == len(hidden_dim) == num_layers:
			raise ValueError('Inconsistent list length.')

		self.height, self.width = input_size

		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.kernel_size = kernel_size
		self.num_layers = num_layers
		self.p_TD = p_TD
		self.batch_first = batch_first
		self.bias = bias
		self.return_all_layers = return_all_layers

		cell_list = []
		for i in range(self.num_layers):
			cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

			cell_list.append(RBM_Cell(input_size=(self.height, self.width),
			                          input_dim=cur_input_dim,
			                          hidden_dim=self.hidden_dim[i],
			                          kernel_size=self.kernel_size[i],
			                          bias=self.bias,
			                          p_TD=self.p_TD))

		self.cell_list = nn.ModuleList(cell_list)

	def forward(self, input_tensor, hidden_state=None):
		"""

		Parameters
		----------
		input_tensor: todo
			5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
		hidden_state: todo
			None. todo implement stateful

		Returns
		-------
		last_state_list, layer_output
		"""
		if not self.batch_first:
			# (t, b, c, h, w) -> (b, t, c, h, w)
			input_tensor.permute(1, 0, 2, 3, 4)

		# Implement stateful ConvLSTM
		if hidden_state is not None:
			pass
		else:
			hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

		layer_output_list = []
		last_state_list = []

		seq_len = input_tensor.size(1)
		cur_layer_input = input_tensor

		for layer_idx in range(self.num_layers):
			c = hidden_state[layer_idx]
			output_inner = []
			for t in range(seq_len):
				h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
				                                 cur_state=c)
				output_inner.append(h)

			layer_output = torch.stack(output_inner, dim=1)
			cur_layer_input = layer_output

			layer_output_list.append(layer_output)
			last_state_list.append(c)

		if not self.return_all_layers:
			layer_output_list = layer_output_list[-1:]
			last_state_list = last_state_list[-1:]

		return layer_output_list, last_state_list

	def _init_hidden(self, batch_size):
		init_states = []
		for i in range(self.num_layers):
			init_states.append(self.cell_list[i].init_hidden(batch_size))
		return init_states

	@staticmethod
	def _check_kernel_size_consistency(kernel_size):
		if not (isinstance(kernel_size, tuple) or
		        (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
			raise ValueError('`kernel_size` must be tuple or list of tuples')

	@staticmethod
	def _extend_for_multilayer(param, num_layers):
		if not isinstance(param, list):
			param = [param] * num_layers
		return param
