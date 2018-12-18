import tensorflow as tf
import random


class RBMCell(object):

	def __init__(self, shape, filter_size, p_TD):
		self.shape = shape
		self.filter_size = filter_size
		self.p_TD = p_TD

	def __call__(self, inputs, state, hidden_dim, idx, scope=None, reuse=False):
		with tf.variable_scope(scope or type(self).__name__, reuse=reuse):
			rate = random.random()

			c = state
			o = inputs
			_o = tf.stop_gradient(inputs) if rate < self.p_TD else inputs

			T = _conv_linear([_o, c], self.filter_size, hidden_dim, True, scope='convT_' + str(idx), reuse=reuse)
			R = _conv_linear([o], self.filter_size, hidden_dim, True, scope='convR_' + str(idx), reuse=reuse)

			_T = tf.nn.sigmoid(T)
			_R = tf.nn.relu(R)

			new_o = tf.multiply(_T, _R)

			return new_o, _T


class RBM(object):

	def __init__(self, shape, filter_size, num_layers, hidden_dims, jump_stride, p_TD):

		self.shape = shape
		self.filter_size = filter_size
		self.num_layers = num_layers
		self.hidden_dims = hidden_dims
		self.jump_stride = jump_stride
		self.p_TD = p_TD

		self.rbmcell = RBMCell(self.shape, self.filter_size, self.p_TD)

	def __call__(self, inputs, state_list, scope=None, reuse=False):
		with tf.variable_scope(scope or type(self).__name__, reuse=reuse):
			layer_state_list = []
			_inputs = [inputs]

			for idx, layer_hiddim in enumerate(self.hidden_dims):
				o, c = self.rbmcell(_inputs[-1], state_list[idx], layer_hiddim, idx)
				if idx % self.jump_stride == self.jump_stride - 1:
					o = o + _inputs[idx + 1 - self.jump_stride]
				layer_state_list.append(c)
				_inputs.append(o)

			return o, layer_state_list


def _conv_linear(args, filter_size, num_features, bias,
                 bias_start=0.0, scope=None, reuse=False):
	"""convolution:
	Args:
	  args: a 4D Tensor or a list of 4D, batch x n, Tensors.
	  filter_size: int tuple of filter height and width.
	  num_features: int, number of features.
	  bias_start: starting value to initialize the bias; 0 by default.
	  scope: VariableScope for the created subgraph; defaults to "Linear".
	  reuse: For reusing already existing weights
	Returns:
	  A 4D Tensor with shape [batch h w num_features]
	Raises:
	  ValueError: if some of the arguments has unspecified or wrong shape.
	"""

	# Calculate the total size of arguments on dimension 1.
	total_arg_size_depth = 0
	shapes = [a.get_shape().as_list() for a in args]
	for shape in shapes:
		if len(shape) != 4:
			raise ValueError("Linear is expecting 4D arguments: %s" % str(shapes))
		if not shape[3]:
			raise ValueError("Linear expects shape[4] of arguments: %s" % str(shapes))
		else:
			total_arg_size_depth += shape[3]

	dtype = [a.dtype for a in args][0]

	# Now the computation.
	with tf.variable_scope(scope or "Conv", reuse=reuse):
		matrix = tf.get_variable(
			"Matrix", [filter_size[0], filter_size[1],
			           total_arg_size_depth, num_features], dtype=dtype)
		if len(args) == 1:
			res = tf.nn.conv2d(args[0], matrix, strides=[1, 1, 1, 1], padding='SAME')
		else:
			res = tf.nn.conv2d(tf.concat(axis=3, values=args), matrix,
			                   strides=[1, 1, 1, 1], padding='SAME')
		if not bias:
			return res
		bias_term = tf.get_variable(
			"Bias", [num_features],
			dtype=dtype, initializer=tf.constant_initializer(bias_start, dtype=dtype)
		)

	return res + bias_term
