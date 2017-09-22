import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.dataset import concat_examples

from utils import get_mnist

class MLP(Chain):
    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)    # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y

def run(train_iter, val_iter, test_data, model, optimizer, max_epoch):
    training_losses = []
    validation_losses = []

    while train_iter.epoch < max_epoch:
        # Get next mini-batch
        batch = train_iter.next()
        image_train, target_train = concat_examples(batch)

        # Prediction
        prediction_train = model(image_train)

        # Compute loss
        loss = F.softmax_cross_entropy(prediction_train, target_train)

        # Compute gradients
        model.cleargrads()
        loss.backward()

        # Update variables
        optimizer.update()

        # Check the validation accuracy of prediction after every epoch
        if train_iter.is_new_epoch:  # If this iteration is the final iteration of the current epoch

            # Display the training loss
            print('epoch:{:02d} train_loss:{:.04f} '.format(train_iter.epoch, float(loss.data)))
            training_losses.append(float(loss.data))

            val_losses = []
            val_accuracies = []
            while True:
                val_batch = val_iter.next()
                image_val, target_val = concat_examples(val_batch)

                # Forward the validation data
                prediction_val = model(image_val)

                # Calculate the loss
                loss_val = F.softmax_cross_entropy(prediction_val, target_val)
                val_losses.append(loss_val.data)

                # Calculate the accuracy
                accuracy = F.accuracy(prediction_val, target_val)
                val_accuracies.append(accuracy.data)

                if val_iter.is_new_epoch:
                    val_iter.epoch = 0
                    val_iter.current_position = 0
                    val_iter.is_new_epoch = False
                    val_iter._pushed_position = None

                    validation_losses.append(np.mean(val_losses))
                    break

            print('val_loss:{:.04f} val_accuracy:{:.04f}'.format(np.mean(val_losses), np.mean(val_accuracies)))

    # Predict full test set
    image_test, target_test = concat_examples(test_data)
    # Forward test data
    prediction_test = model(image_test)
    # Calculate loss and accuracy
    loss_test = F.softmax_cross_entropy(prediction_test, target_test)
    accuracy_test = F.accuracy(prediction_test, target_test)

    print('test_loss: ' + str(loss_test.data) + ' test_accuracy: ' + str(accuracy_test.data))
    return training_losses, validation_losses


def main():
    # Load data
    train, test = get_mnist()
    
    # Initialize iterators
    train_iter = iterators.SerialIterator(train, batch_size=32, shuffle=True)
    val_iter = iterators.SerialIterator(test, batch_size=50, repeat=False, shuffle=False)

    # Define model
    model = MLP(10, 10)
    optimizer = optimizers.SGD()
    optimizer.setup(model)

    training_losses, validation_losses = run(train_iter, val_iter, test, model, optimizer, 200)

if __name__ == "__main__":
    main()