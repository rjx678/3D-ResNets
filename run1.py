from datetime import datetime
import time
from ResNet import ResNet_3D
from utils1 import *
import os
import input_data_reads

rootPath = r'################'
os.chdir(rootPath)
now = datetime.now()
logs_path = "./graph50/" + now.strftime("%Y%m%d-%H%M%S")
save_dir = "./checkpoints50/"

image_size = 224
num_classes = 3
num_channels = 3
num_epochs=10
display=10
num_steps = 5000
batch_size = 6
# Creating the ResNet model
model = ResNet_3D(num_classes, image_size, num_channels)
model.inference().pred_func().accuracy_func().loss_func().train_func()
# Saving the best trained model (based on the validation accuracy)
saver = tf.train.Saver()
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'best_validation')
best_validation_accuracy = 0

acc_b_all = acc_v_all=mean_v_acc=mean_acc = loss_v_all=mean_v_loss=loss_b_all = mean_loss = np.array([])
sum_count = 0
starttime=time.time()
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    print("Initialized")
    merged = tf.summary.merge_all()
    batch_writer = tf.summary.FileWriter(logs_path + '/train/', sess.graph)
    valid_writer = tf.summary.FileWriter(logs_path + '/test/',sess.graph)

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print('-----------------------------------------------------------------------------')
        print('Epoch: {}'.format(epoch+1))

        for step in range(num_steps):
            train_images, train_labels = input_data_reads.read_clip_and_label(
                filename='./c3d_dataset/train.list',
                batch_size=6,
                num_frames_per_clip=6,
                crop_size=224,
                shuffle=True
            )

            X_train, Y_train = train_images, train_labels
            _ = sess.run(model.train_op,feed_dict={model.x:X_train,model.y:Y_train,model.keep_prob:0.5})

            if step % display == 0:
                feed_dict_batch = {model.x: X_train, model.y: Y_train, model.keep_prob: 1.0}
                acc_b, loss_b = sess.run([model.accuracy, model.loss], feed_dict=feed_dict_batch)
                acc_b_all = np.append(acc_b_all, acc_b)
                loss_b_all = np.append(loss_b_all, loss_b)
                mean_acc = np.mean(acc_b_all)
                mean_loss = np.mean(loss_b_all)
                print("Step {0}, training loss: {1:.5f}, training accuracy: {2:.05%}".format(step, mean_loss, mean_acc))

                summary_tr = tf.Summary(value=[tf.Summary.Value(tag='Accuracy', simple_value=mean_acc)])
                batch_writer.add_summary(summary_tr, sum_count * display)
                summary_tr = tf.Summary(value=[tf.Summary.Value(tag='Loss', simple_value=mean_loss)])
                batch_writer.add_summary(summary_tr, sum_count * display)
                summary = sess.run(merged, feed_dict=feed_dict_batch)
                batch_writer.add_summary(summary, sum_count * display)

                val_images, val_labels = input_data_reads.read_clip_and_label(
                    filename='./c3d_dataset/test.list',
                    batch_size=6,
                    num_frames_per_clip=6,
                    crop_size=224,
                    shuffle=True
                )
                feed_dict_val = {model.x: val_images, model.y: val_labels, model.keep_prob: 1.0}
                acc_valid, loss_valid = sess.run([model.accuracy, model.loss], feed_dict=feed_dict_val)
                acc_v_all = np.append(acc_v_all, acc_valid)
                loss_v_all = np.append(loss_v_all,loss_valid)
                mean_v_acc=np.mean(acc_v_all)
                mean_v_loss=np.mean(loss_v_all)
                print("Step {0}, test mean acc: {1:.05%}".format(step,mean_v_acc))

                summary_valid = tf.Summary(value=[tf.Summary.Value(tag='Accuracy', simple_value=mean_v_acc)])
                valid_writer.add_summary(summary_valid, sum_count * display)
                summary_v = sess.run(merged, feed_dict=feed_dict_val)
                valid_writer.add_summary(summary_v, sum_count * display)
                if mean_v_acc > best_validation_accuracy:
                    # Update the best-known validation accuracy.
                    best_validation_accuracy = mean_v_acc
                    best_epoch = epoch
                    # Save all variables of the TensorFlow graph to file.
                    saver.save(sess=sess, save_path=save_path)
                    # A string to be printed below, shows improvement found.
                    improved_str = '*'
                    print("Epoch {0}, test mean acc: {1:.05f}{2}"
                          .format(epoch + 1, mean_v_acc, improved_str))
                else:
                    # An empty string to be printed below.
                    # Shows that no improvement was found.
                    improved_str = ''

                sum_count += 1
                # acc_b_all =loss_b_all=  mean_acc = mean_loss = y_pred_all = y_true_all = np.array([])

print((time.time()-starttime)/3600)# 18 6.97hour
