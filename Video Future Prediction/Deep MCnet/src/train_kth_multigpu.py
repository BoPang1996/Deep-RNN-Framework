import time
import tensorflow as tf

from mcnet import MCNET
from utils import *
from os import makedirs
from os.path import exists
from argparse import ArgumentParser
from joblib import Parallel, delayed


def average_gradients(tower_grads):
	average_grads = []

	for grad_and_vars in zip(*tower_grads):
		# Note that each grad_and_vars looks like the following:
		#   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
		grads = []
		for g, _ in grad_and_vars:
			# Add 0 dimension to the gradients to represent the tower.
			expanded_g = tf.expand_dims(g, 0)

			# Append on a 'tower' dimension which we will average over below.
			grads.append(expanded_g)

		# Average over the 'tower' dimension.
		grad = tf.concat(axis=0, values=grads)
		grad = tf.reduce_mean(grad, 0)

		# Keep in mind that the Variables are redundant because they are shared
		# across towers. So .. we will just return the first tower's pointer to
		# the Variable.
		v = grad_and_vars[0][1]
		grad_and_var = (grad, v)
		average_grads.append(grad_and_var)

	return average_grads


def train(lr, batch_size, alpha, beta, image_size, K, T, num_iter, gpu):
	with tf.Graph().as_default(), tf.device('/cpu:0'):
		num_gpus = len(gpu)
		data_path = "../data/KTH/"
		f = open(data_path + "train_data_list_trimmed.txt", "r")
		trainfiles = f.readlines()

		margin = 0.3
		updateD = True
		updateG = True
		iters = 0

		gpu_prefix = ""
		for gpu_id in gpu:
			gpu_prefix += str(gpu_id)

		prefix = ("KTH_MCNET"
		          + "_gpu_id=" + gpu_prefix
		          + "_image_size=" + str(image_size)
		          + "_K=" + str(K)
		          + "_T=" + str(T)
		          + "_batch_size=" + str(batch_size)
		          + "_alpha=" + str(alpha)
		          + "_beta=" + str(beta)
		          + "_lr=" + str(lr)
		          + "_num_layer=" + str(15))

		print("\n" + prefix + "\n")
		checkpoint_dir = "../models/" + prefix + "/"
		summary_dir = "../logs/" + prefix + "/"

		if not exists(checkpoint_dir):
			makedirs(checkpoint_dir)
		if not exists(summary_dir):
			makedirs(summary_dir)

		global_step = tf.get_variable(
			'global_step', [],
			initializer=tf.constant_initializer(0), trainable=False)

		model = MCNET(image_size=[image_size, image_size], c_dim=1, K=K,
		              batch_size=batch_size, T=T, checkpoint_dir=checkpoint_dir)

		d_optim = tf.train.AdamOptimizer(lr, beta1=0.5)
		g_optim = tf.train.AdamOptimizer(lr, beta1=0.5)

		_diff_in = tf.placeholder(tf.float32, [batch_size * num_gpus, image_size, image_size, K - 1, 1])
		_xt = tf.placeholder(tf.float32, [batch_size * num_gpus, image_size, image_size, model.c_dim])
		_target = tf.placeholder(tf.float32, [batch_size * num_gpus, image_size, image_size, K + T, model.c_dim])

		d_tower_grads = []
		g_tower_grads = []

		avg_errD_fake = []
		avg_errD_real = []
		avg_errG = []

		with tf.variable_scope(tf.get_variable_scope()):

			for i in range(num_gpus):
				with tf.device('/gpu:%d' % gpu[i]):
					with tf.name_scope('tower_%d' % gpu[i]) as scope:
						model.build_model(_diff_in[batch_size * i: batch_size * (i + 1), :, :, :, :],
						                  _xt[batch_size * i: batch_size * (i + 1), :, :, :],
						                  _target[batch_size * i: batch_size * (i + 1), :, :, :, :])
						DES_LOSS = model.d_loss
						GEN_LOSS = alpha * model.L_img + beta * model.L_GAN

						avg_errD_fake.append(model.d_loss_fake)
						avg_errD_real.append(model.d_loss_real)
						avg_errG.append(model.L_GAN)

						def tower_loss_D():
							D_total_loss = tf.constant(0)

							tf.add_to_collection('D_LOSS', DES_LOSS)
							D_losses = tf.get_collection('D_LOSS', scope)
							D_total_loss = tf.add_n(D_losses, name='D_total_loss')

							return D_total_loss

						def tower_loss_G():
							G_total_loss = tf.constant(0)

							tf.add_to_collection('G_LOSS', GEN_LOSS)
							G_losses = tf.get_collection('G_LOSS', scope)
							G_total_loss = tf.add_n(G_losses, name='G_total_loss')

							return G_total_loss

						loss_D = tower_loss_D()
						loss_G = tower_loss_G()

						tf.get_variable_scope().reuse_variables()

						d_grads = d_optim.compute_gradients(loss_D, var_list=model.d_vars)
						d_tower_grads.append(d_grads)

						g_grads = g_optim.compute_gradients(loss_G, var_list=model.g_vars)
						g_tower_grads.append(g_grads)

		d_grads = average_gradients(d_tower_grads)
		d_apply_gradient_op = d_optim.apply_gradients(d_grads, global_step=global_step)

		g_grads = average_gradients(g_tower_grads)
		g_apply_gradient_op = g_optim.apply_gradients(g_grads, global_step=global_step)

		errD_fake_op = tf.add_n(avg_errD_fake) / float(len(avg_errD_fake))
		errD_real_op = tf.add_n(avg_errD_real) / float(len(avg_errD_real))
		errG_op = tf.add_n(avg_errG) / float(len(avg_errG))

		summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
		summary_op = tf.summary.merge(summaries)

		# =============================================== TRAINING ================================================== #

		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
		config = tf.ConfigProto(allow_soft_placement=True,
		                        log_device_placement=False,
		                        gpu_options=gpu_options)
		config.gpu_options.allow_growth = True
		with tf.Session(config=config) as sess:
			tf.global_variables_initializer().run()

			if model.load(sess, checkpoint_dir):
				print(" [*] Load SUCCESS")
			else:
				print(" [!] Load failed...")

			writer = tf.summary.FileWriter(summary_dir, sess.graph)

			counter = iters + 1
			start_time = time.time()

			while iters < num_iter:
				mini_batches = get_minibatches_idx(len(trainfiles), batch_size * num_gpus, shuffle=True)
				with Parallel(n_jobs=batch_size * num_gpus) as parallel:
					for _, batchidx in mini_batches:
						if len(batchidx) == batch_size * num_gpus:
							seq_batch = np.zeros((batch_size * num_gpus, image_size, image_size,
							                      K + T, 1), dtype="float32")
							diff_batch = np.zeros((batch_size * num_gpus, image_size, image_size,
							                       K - 1, 1), dtype="float32")
							t0 = time.time()
							Ts = np.repeat(np.array([T]), batch_size * num_gpus, axis=0)
							Ks = np.repeat(np.array([K]), batch_size * num_gpus, axis=0)
							paths = np.repeat(data_path, batch_size * num_gpus, axis=0)
							tfiles = np.array(trainfiles)[batchidx]
							shapes = np.repeat(np.array([image_size]), batch_size * num_gpus, axis=0)
							output = parallel(delayed(load_kth_data)(f, p, img_sze, k, t)
							                  for f, p, img_sze, k, t in zip(tfiles, paths, shapes, Ks, Ts))

							for i in range(batch_size * num_gpus):
								seq_batch[i] = output[i][0]
								diff_batch[i] = output[i][1]

							if updateD:
								_, summary_str = sess.run([d_apply_gradient_op, summary_op], feed_dict={
									_diff_in: diff_batch,
									_xt: seq_batch[:, :, :, K - 1],
									_target: seq_batch
								})
							if updateG:
								_, summary_str = sess.run([g_apply_gradient_op, summary_op], feed_dict={
									_diff_in: diff_batch,
									_xt: seq_batch[:, :, :, K - 1],
									_target: seq_batch
								})

							writer.add_summary(summary_str, counter)

							errD_fake, errD_real, errG = sess.run([errD_fake_op, errD_real_op, errG_op],
							                                      feed_dict={
								                                      _diff_in: diff_batch,
								                                      _xt: seq_batch[:, :, :, K - 1],
								                                      _target: seq_batch
							                                      })

							if errD_fake < margin or errD_real < margin:
								updateD = False
							if errD_fake > (1. - margin) or errD_real > (1. - margin):
								updateG = False
							if not updateD and not updateG:
								updateD = True
								updateG = True

							counter += 1

							print(
								"Iters: [%2d] time: %4.4f, d_loss: %.8f, L_GAN: %.8f"
								% (iters, time.time() - start_time, errD_fake + errD_real, errG)
							)

							if np.mod(counter, 250) == 2:
								model.save(sess, checkpoint_dir, counter)

							iters += 1


if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("--lr", type=float, dest="lr",
	                    default=0.0001, help="Base Learning Rate")
	parser.add_argument("--batch_size", type=int, dest="batch_size",
	                    default=8, help="Mini-batch size")
	parser.add_argument("--alpha", type=float, dest="alpha",
	                    default=1.0, help="Image loss weight")
	parser.add_argument("--beta", type=float, dest="beta",
	                    default=0.02, help="GAN loss weight")
	parser.add_argument("--image_size", type=int, dest="image_size",
	                    default=128, help="Mini-batch size")
	parser.add_argument("--K", type=int, dest="K",
	                    default=10, help="Number of steps to observe from the past")
	parser.add_argument("--T", type=int, dest="T",
	                    default=10, help="Number of steps into the future")
	parser.add_argument("--num_iter", type=int, dest="num_iter",
	                    default=30000, help="Number of iterations")
	parser.add_argument("--gpu", type=int, nargs="+", dest="gpu", required=True,
	                    help="GPU device id")

	args = parser.parse_args()
	train(**vars(args))
