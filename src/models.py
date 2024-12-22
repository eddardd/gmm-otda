import torch
import pytorch_lightning as pl
from torchmetrics.functional import accuracy


class ShallowNeuralNet(pl.LightningModule):
    r"""Shallow Neural Network model with Pytorch Lightning

    This class implements a single layer neural network which
    predicts a classes from feature vectors.

    Parameters
    ----------
    n_features : int
        Number of features in the input of the network.
    n_classes : int
        Number of classes in the output of the network.
    learning_rate : float, optional (default=None)
        Learning rate for the classifier part of the network.
        If it is None, uses 10 * learning_rate_encoder.
    loss_fn : function, optional (default=None)
        Function taking as arguments the network predictions
        and the ground-truths, and returns a differentiable
        loss. If None, uses torch.nn.CrossEntropyLoss().
    l2_penalty : float, optional (default=0.0)
        If positive, adds l2_penalty to the network weights.
    weight_decay : float, optional (default=0.0)
        If positive, adds weight decay to network weights.
    momentum : float, optional (default=0.9)
        Only used if optimizer is SGD. Momentum term in the optimizer.
    log_gradients : bool, optional (default=False)
        If True, logs histograms of network gradients. NOTE: the
        generated tensorboard logs may be heavy if this parameter
        is set to true.
    optimizer_name : str, optional (default='adam')
        Either 'adam' or 'sgd'. Chooses which optimization strategy
        the network will adopt.
    max_norm : float, optional (default=None)
        If given, constrains the network weights to have maximum norm
        equal to the given value.
    """
    def __init__(self,
                 n_features,
                 n_classes,
                 learning_rate=1e-4,
                 loss_fn=None,
                 l2_penalty=0.0,
                 momentum=0.9,
                 optimizer_name='adam',
                 log_gradients=False,
                 max_norm=None):
        super(ShallowNeuralNet, self).__init__()
        self.main = torch.nn.Linear(n_features, n_classes)
        self.learning_rate = learning_rate

        if loss_fn is None:
            self.loss_fn = torch.nn.CrossEntropyLoss()
        else:
            self.loss_fn = loss_fn

        self.l2_penalty = l2_penalty
        self.momentum = momentum
        self.history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}
        self.log_gradients = log_gradients
        self.optimizer_name = optimizer_name
        self.n_classes = n_classes
        self.max_norm = max_norm

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def max_norm_normalization(self, w):
        with torch.no_grad():
            norm = torch.sqrt(torch.sum(w ** 2, dim=1, keepdim=True))
            desired = torch.clamp(norm, 0, self.max_norm)
            w *= (desired / (1e-10 + norm))

    def custom_histogram_adder(self):
        if self.logger is not None:
            for name, params in self.named_parameters():
                self.logger.experiment.add_histogram(name,
                                                     params,
                                                     self.current_epoch)

    def forward(self, x):
        if self.max_norm is not None:
            self.max_norm_normalization(self.main.weight)
        return self.main(x)

    def configure_optimizers(self):
        if self.optimizer_name.lower() == 'adam':
            return torch.optim.Adam(self.parameters(),
                                    lr=self.learning_rate,
                                    weight_decay=self.l2_penalty)
        else:
            return torch.optim.SGD(self.parameters(),
                                   lr=self.learning_rate,
                                   weight_decay=self.l2_penalty,
                                   momentum=self.momentum)

    def __step(self, batch, batch_idx):
        x, y = batch

        y_pred = self(x)

        L = self.loss_fn(target=y.argmax(dim=1), input=y_pred)
        acc = accuracy(preds=y_pred,
                       target=y.argmax(dim=1),
                       task='multiclass',
                       num_classes=self.n_classes,
                       top_k=1)

        return {'loss': L, 'acc': acc}

    def training_step(self, batch, batch_idx):
        output = self.__step(batch, batch_idx)
        self.training_step_outputs.append(output)
        return output

    def validation_step(self, batch, batch_idx):
        output = self.__step(batch, batch_idx)
        self.validation_step_outputs.append(output)

        return output

    def test_step(self, batch, batch_idx):
        output = self.__step(batch, batch_idx)
        self.test_step_outputs.append(output)

        return output

    def on_train_epoch_end(self):
        avg_loss = torch.tensor([
            x['loss'] for x in self.training_step_outputs]).mean()
        avg_acc = torch.tensor([
            x['acc'] for x in self.training_step_outputs]).mean()
        self.training_step_outputs.clear()

        self.log('loss', avg_loss)
        self.log('accuracy', avg_acc)

        self.history['loss'].append(avg_loss)
        self.history['acc'].append(avg_acc)

        # Logs scalars
        if self.logger is not None:
            self.logger.experiment.add_scalar("Loss/Train",
                                              avg_loss,
                                              self.current_epoch)
            self.logger.experiment.add_scalar("Accuracy/Train",
                                              avg_acc,
                                              self.current_epoch)

        # Logs histograms
        if self.log_gradients:
            self.custom_histogram_adder()

    def on_validation_epoch_end(self):
        avg_loss = torch.tensor([
            x['loss'] for x in self.validation_step_outputs]).mean()
        avg_acc = torch.tensor([
            x['acc'] for x in self.validation_step_outputs]).mean()
        self.validation_step_outputs.clear()

        self.log('val_loss', avg_loss)
        self.log('val_accuracy', avg_acc)

        self.history['val_loss'].append(avg_loss)
        self.history['val_acc'].append(avg_acc)

        if self.logger is not None:
            self.logger.experiment.add_scalar("Loss/Validation",
                                              avg_loss,
                                              self.current_epoch)
            self.logger.experiment.add_scalar("Accuracy/Validation",
                                              avg_acc,
                                              self.current_epoch)

    def on_test_epoch_end(self):
        avg_loss = torch.tensor([
            x['loss'] for x in self.test_step_outputs]).mean()
        avg_acc = torch.tensor([
            x['acc'] for x in self.test_step_outputs]).mean()
        self.test_step_outputs.clear()

        if self.logger is not None:
            self.logger.experiment.add_scalar("Loss/Test",
                                              avg_loss,
                                              self.current_epoch)
            self.logger.experiment.add_scalar("Accuracy/Test",
                                              avg_acc,
                                              self.current_epoch)


class WeightedShallowNeuralNet(pl.LightningModule):
    r"""Weighted Shallow Neural Network model with Pytorch Lightning

    This class implements a single layer neural network which
    predicts a classes from feature vectors. Losses can be weighted
    by sample importance

    Parameters
    ----------
    n_features : int
        Number of features in the input of the network.
    n_classes : int
        Number of classes in the output of the network.
    learning_rate : float, optional (default=None)
        Learning rate for the classifier part of the network.
        If it is None, uses 10 * learning_rate_encoder.
    loss_fn : function, optional (default=None)
        Function taking as arguments the network predictions
        and the ground-truths, and returns a differentiable
        loss. If None, uses torch.nn.CrossEntropyLoss().
    l2_penalty : float, optional (default=0.0)
        If positive, adds l2_penalty to the network weights.
    weight_decay : float, optional (default=0.0)
        If positive, adds weight decay to network weights.
    momentum : float, optional (default=0.9)
        Only used if optimizer is SGD. Momentum term in the optimizer.
    log_gradients : bool, optional (default=False)
        If True, logs histograms of network gradients. NOTE: the
        generated tensorboard logs may be heavy if this parameter
        is set to true.
    optimizer_name : str, optional (default='adam')
        Either 'adam' or 'sgd'. Chooses which optimization strategy
        the network will adopt.
    max_norm : float, optional (default=None)
        If given, constrains the network weights to have maximum norm
        equal to the given value.
    """
    def __init__(self,
                 n_features,
                 n_classes,
                 learning_rate=1e-4,
                 l2_penalty=0.0,
                 momentum=0.9,
                 optimizer_name='adam',
                 log_gradients=False,
                 max_norm=None):
        super(WeightedShallowNeuralNet, self).__init__()
        self.main = torch.nn.Linear(n_features, n_classes)
        self.learning_rate = learning_rate

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

        self.l2_penalty = l2_penalty
        self.momentum = momentum
        self.history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}
        self.log_gradients = log_gradients
        self.optimizer_name = optimizer_name
        self.n_classes = n_classes
        self.max_norm = max_norm

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def max_norm_normalization(self, w):
        with torch.no_grad():
            norm = torch.sqrt(torch.sum(w ** 2, dim=1, keepdim=True))
            desired = torch.clamp(norm, 0, self.max_norm)
            w *= (desired / (1e-10 + norm))

    def custom_histogram_adder(self):
        if self.logger is not None:
            for name, params in self.named_parameters():
                self.logger.experiment.add_histogram(name,
                                                     params,
                                                     self.current_epoch)

    def forward(self, x):
        if self.max_norm is not None:
            self.max_norm_normalization(self.main.weight)
        return self.main(x)

    def configure_optimizers(self):
        if self.optimizer_name.lower() == 'adam':
            return torch.optim.Adam(self.parameters(),
                                    lr=self.learning_rate,
                                    weight_decay=self.l2_penalty)
        else:
            return torch.optim.SGD(self.parameters(),
                                   lr=self.learning_rate,
                                   weight_decay=self.l2_penalty,
                                   momentum=self.momentum)

    def __step(self, batch, batch_idx):
        w, x, y = batch

        y_pred = self(x)

        # NOTE. L is a vector of shape (n,)
        Lvec = self.loss_fn(target=y.argmax(dim=1), input=y_pred)
        # We need to multiply it by the sample importance vector
        L = (Lvec * (w / (w.sum() + 1e-9))).sum()
        acc = accuracy(preds=y_pred,
                       target=y.argmax(dim=1),
                       task='multiclass',
                       num_classes=self.n_classes,
                       top_k=1)

        return {'loss': L, 'acc': acc}

    def training_step(self, batch, batch_idx):
        output = self.__step(batch, batch_idx)
        self.training_step_outputs.append(output)
        return output

    def validation_step(self, batch, batch_idx):
        output = self.__step(batch, batch_idx)
        self.validation_step_outputs.append(output)

        return output

    def test_step(self, batch, batch_idx):
        output = self.__step(batch, batch_idx)
        self.test_step_outputs.append(output)

        return output

    def on_train_epoch_end(self):
        avg_loss = torch.tensor([
            x['loss'] for x in self.training_step_outputs]).mean()
        avg_acc = torch.tensor([
            x['acc'] for x in self.training_step_outputs]).mean()
        self.training_step_outputs.clear()

        self.log('loss', avg_loss)
        self.log('accuracy', avg_acc)

        self.history['loss'].append(avg_loss)
        self.history['acc'].append(avg_acc)

        # Logs scalars
        if self.logger is not None:
            self.logger.experiment.add_scalar("Loss/Train",
                                              avg_loss,
                                              self.current_epoch)
            self.logger.experiment.add_scalar("Accuracy/Train",
                                              avg_acc,
                                              self.current_epoch)

        # Logs histograms
        if self.log_gradients:
            self.custom_histogram_adder()

    def on_validation_epoch_end(self):
        avg_loss = torch.tensor([
            x['loss'] for x in self.validation_step_outputs]).mean()
        avg_acc = torch.tensor([
            x['acc'] for x in self.validation_step_outputs]).mean()
        self.validation_step_outputs.clear()

        self.log('val_loss', avg_loss)
        self.log('val_accuracy', avg_acc)

        self.history['val_loss'].append(avg_loss)
        self.history['val_acc'].append(avg_acc)

        if self.logger is not None:
            self.logger.experiment.add_scalar("Loss/Validation",
                                              avg_loss,
                                              self.current_epoch)
            self.logger.experiment.add_scalar("Accuracy/Validation",
                                              avg_acc,
                                              self.current_epoch)

    def on_test_epoch_end(self):
        avg_loss = torch.tensor([
            x['loss'] for x in self.test_step_outputs]).mean()
        avg_acc = torch.tensor([
            x['acc'] for x in self.test_step_outputs]).mean()
        self.test_step_outputs.clear()

        if self.logger is not None:
            self.logger.experiment.add_scalar("Loss/Test",
                                              avg_loss,
                                              self.current_epoch)
            self.logger.experiment.add_scalar("Accuracy/Test",
                                              avg_acc,
                                              self.current_epoch)
