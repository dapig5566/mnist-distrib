import tensorflow.compat.v1 as tf
import numpy as np
from tensorflow.python.platform import flags
from loader import DataManager
from models import Classifier

FLAGS = flags.FLAGS
flags.DEFINE_string("job_name", "", "the job name.")
flags.DEFINE_integer("task_id", 0, "the task id.")
flags.DEFINE_integer("epochs", 200, "epochs")
flags.DEFINE_integer("batch_size", 256, "the batch size.")
flags.DEFINE_float("lr", 1e-3, "the learning rate.")
tf.disable_v2_behavior()
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True
cluster = tf.train.ClusterSpec({"ps": ["localhost:2222"], "worker": ["localhost:2223", "localhost:2224"]})
server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_id, config=config)

if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":

    is_chief = FLAGS.task_id == 0
    def preprocess(image, label):
        image = (image - dm.mean) / dm.std
        label = tf.one_hot(label, dm.num_class,)
        return image, label

    def setup_dataset():
        train_input = tf.placeholder(tf.float32, [None, dm.w, dm.h, dm.c])
        train_label = tf.placeholder(tf.int32, [None])
        dataset = tf.data.Dataset.from_tensor_slices((train_input, train_label))
        dataset = dataset.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(10000)
        dataset = dataset.batch(FLAGS.batch_size).repeat(10)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        iterator = dataset.make_initializable_iterator()
        next_batch = iterator.get_next()
        return [train_input, train_label], next_batch, iterator.initializer

    dm = DataManager("datasets")
    dm.load_MNIST(NHWC=True)
    iterations = dm.train_label.shape[0] // FLAGS.batch_size + 1
    data, next_batch, init = setup_dataset()
    with tf.device(tf.train.replica_device_setter(ps_device="/job:ps/task:0/device:GPU:0", worker_device="/job:worker/task:{0}/device:GPU:{0}".format(FLAGS.task_id), cluster=cluster)):
        model = Classifier(dm.num_class)
        next_input, next_label = next_batch
        logits = model.build_convnet(next_input)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=next_label, axis=-1))
        acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(logits, axis=-1), tf.argmax(next_label, axis=-1))))
        opt = tf.train.AdamOptimizer(FLAGS.lr)
        opt = tf.train.SyncReplicasOptimizer(opt, 2, 2)
        global_step = tf.train.get_or_create_global_step()
        train_op = opt.minimize(loss, global_step)
        sync_replicas_hook = opt.make_session_run_hook(is_chief)
        # step_hook = tf.train.StopAtStepHook(last_step=40000)
        hooks = [sync_replicas_hook]

    with tf.train.MonitoredTrainingSession(master=server.target, is_chief=is_chief, hooks=hooks) as ss:
        for i in range(FLAGS.epochs):
            if i % 10 == 0:
                ss.run(init, feed_dict=dict(zip(data, [dm.train_input, dm.train_label])))
            for j in range(iterations):
                _, loss_value, acc_value = ss.run([train_op, loss, acc])
                if j % 20 == 0:
                    print("ep [{}/{}], iter [{}/{}]: loss/acc: {:.4f}/{:.4f}".format(i, FLAGS.epochs, j, iterations, loss_value, acc_value))
else:
    raise ValueError("not recognized job name.")
