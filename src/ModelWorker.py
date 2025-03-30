import torch
from tqdm import tqdm

class ModelWorker:
    """
    General class to run training, validation, testing loops and to provide desired metrics for a model.

    Args:
        - train_data: Dataloader for the training data images and labels.
        - validation_data: Dataloader for the validation data images and labels.
        - loss_function: Loss function for the model
        - optimizer: Optimizer for the model.
        - model: Model to train

    To use, call the 'model_train_val' function for training a model, then call 'model_test' for evaluations.
    """

    def __init__(self,
                 train_data: torch.utils.data.DataLoader,
                 validation_data: torch.utils.data.DataLoader,
                 loss_function: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 model: torch.nn.Module):

        self._train_data = train_data
        self._validation_data = validation_data
        self._loss_function = loss_function
        self._optimizer = optimizer
        self._model = model

        if torch.cuda.is_available():
            self._device = 'cuda'
        else:
            self._device = 'cpu'

        print(f"Pytorch found device: {self._device}")

        # Send the components to whatever device is currently available.
        self._model.to(self._device)
        self._loss_function.to(self._device)

        # Store general metrics
        self._train_metrics = {'LOSS': None}
        self._validation_metrics = {'LOSS': None}
        self._test_metrics = {'LOSS': None}

    def model_train_val(self, epochs: int, quiet: bool = True):
        """
        Train and Validate the given model for this worker on the given dataset.

        Args:
            - epochs: Number of epochs to run training and validation for.
            - quiet: Whether to silence print statements.
        """
        # Set the model to train mode.
        self._model.train()

        train_losses = []
        validation_losses = []

        for epoch in range(epochs):
            if not quiet:
                print(f"Epoch: {epoch}:")

            train_losses.append(self.__train__(quiet))
            validation_losses.append(self.__validation__(quiet))

        self._train_metrics['LOSS'] = train_losses
        self._validation_metrics['LOSS'] = validation_losses

    def __train__(self, quiet: bool = True):
        """
        Run Training step.  Not to be called directly.

        Args:
            - quiet: Whether to silence print statements.
        """
        train_epoch_loss = 0

        for batch_iteration, (images, labels) in enumerate(self._train_data):

            # Send the Images and Labels to the same device as the model.
            images = images.to(self._device)
            labels = labels.to(self._device)

            # Forward Propagation
            predictions = self._model(images)

            # Calculate Loss
            train_loss = self._loss_function(predictions, labels)

            # Backpropagation
            self._optimizer.zero_grad()
            train_loss.backward()
            self._optimizer.step()

            train_epoch_loss += train_loss.item()

            if not quiet:
                print(f"\tTrain Batch Number: {batch_iteration}")

        if not quiet:
            print(f"\tTraining Loss: {train_epoch_loss}")

        return train_epoch_loss

    def __validation__(self, quiet: bool = True):
        """
        Run Validation step.  Not to be called directly.

        Args:
            - quiet: Whether to allow print statements or not.
        """
        validation_epoch_loss = 0

        # Run validation batch
        for batch_iteration, (images, labels) in enumerate(self._validation_data):
            images = images.to(self._device)
            labels = labels.to(self._device)

            predictions = self._model(images)

            validation_loss = self._loss_function(predictions, labels)

            validation_epoch_loss += validation_loss.item()

            if not quiet:
                print(f"\tValidation Batch Number: {batch_iteration}")

        if not quiet:
            print(f"\tValidation Loss: {validation_epoch_loss}")

        return validation_epoch_loss


    def model_test(self, test_data: torch.utils.data.DataLoader, quiet: bool = True):
        """
        Run Testing on the given model for this worker.

        Args:
            - test_data: Loader object for all testing data.
            - quiet: Whether to silence print statements.
        """
        # Set the model to evaluation mode
        self._model.eval()

        testing_losses = []

        with torch.no_grad():

            for test_batch, (image, label) in enumerate(test_data):
                image = image.to(self._device)
                label = label.to(self._device)

                predictions = self._model(image)

                test_loss = self._loss_function(predictions, label)

                testing_losses.append(test_loss.item())

                if not quiet:
                    print(f"Test Batch: {test_batch}")
                    print(f"\tTest Loss: {test_loss.item()}")

        self._test_metrics['LOSS'] = testing_losses


class ModelWorkerFRCNN:
    """
    Training, validation, & testing functionality for use FRCNN in Pytorch.

    Args:
        - train_data: Dataloader for the training data images and labels.
        - validation_data: Dataloader for the validation data images and labels.
        - optimizer: Optimizer for the model.
        - frcnn: FRCNN specific model to train.
        - quiet: Whether to print training, validation, and testing messages.
        - debug: Prints additional messages for debugging throughout code.


    NOTE: FRCNN returns a dictionary of different loss values, each described below:
        1. loss_objectness: Binary cross entropy function to measure whether a region proposal is "background" or an object
        2. loss_rpn_box_reg: L1 Loss function to measure the loss in the region proposal's bounding box coordinates.
        3. loss_classifier: Cross Entropy loss function to measure the the classification loss of an object within a region proposal.
        4. loss_box_reg: L1 Loss function to measure the loss in the bounding box coordinates of the object in a region proposal.
    """
    def __init__(self,
                 train_dataloader: torch.utils.data.DataLoader,
                 validation_dataloader: torch.utils.data.DataLoader,
                 optimizer: torch.optim.Optimizer,
                 frcnn: torch.nn.Module,
                 quiet: bool = True,
                 debug: bool = False):

        self._train_data = train_dataloader
        self._validation_data = validation_dataloader
        self._optimizer = optimizer
        self._frcnn = frcnn
        self._quiet = quiet
        self._debug = debug

        if torch.cuda.is_available():
            self._device = 'cuda'
        else:
            self._device = 'cpu'

        if self._debug:
            print(f"Pytorch found device: {self._device}")

        # Send the components to whatever device is currently available.
        self._frcnn.to(self._device)

        # Store general metrics
        self._train_metrics = {'LOSS': None}
        self._validation_metrics = {'LOSS': None}
        self._test_metrics = {'LOSS': None}

    def train(self, num_epochs: int, indices_to_skip: list = []):
        """
        Run Training step.

        Args:
            - num_epochs: Number of training iterations to run.
            - skip_index: Data instances to skip during training
                - Some data instances are missing their annotations in the Kaggle dataset.  We will skip these and notify the
                  creator.
                - Known Indices: [3736]
        """

        self._frcnn.train()
        train_epoch_loss = 0
        training_losses = []

        for epoch in range(num_epochs):
            for i, (images, annotations) in enumerate(self._train_data):
                if i in indices_to_skip:
                    continue
                try:
                    # Move all images and annotation values to the correct device
                    images = tuple([image.to(self._device) for image in images])
                    annotations = [{key: value.to(self._device) for key, value in target.items()} for target in annotations]

                    # Do forward propagation and get the loss values for the image
                    # NOTE: train_loss is a dictionary containing different loss values for different things.
                    # Summing them produces a general loss value for this image, but we can extract different
                    # metrics using each loss value.  See class description and Pytorch documentation for more
                    # information.
                    train_loss = self._frcnn(images, annotations)
                    train_loss = [loss for loss in train_loss.values()]
                    train_loss = sum(train_loss)

                    # Backpropagation
                    self._optimizer.zero_grad()
                    train_loss.backward()
                    self._optimizer.step()

                    train_epoch_loss += train_loss
                    if not self._quiet or self._debug:
                        print(f"Batch: {i}/{len(self._train_data)} | Batch Loss: {train_loss}\r", end="")
                except:
                    print(f"Failed on Batch {i}")
                    print(f"Image: {images}")
                    print(f"Annotations: {annotations}")

            training_losses.append(train_epoch_loss)
            if not self._quiet:
                print(f"\n\n############# Epoch: {epoch} Complete #############")
                print(f"\t- Epoch Loss: {train_epoch_loss}\n\n")

        self._train_metrics['LOSS'] = training_losses

    def validation(self, num_epochs: int, indices_to_skip: list = []):
        """
        Run Validation step.

        Args:
            - num_epochs: Number of epochs to run validation.
        """
        torch.cuda.empty_cache()
        self._frcnn.train()
        validation_epoch_loss = 0
        validation_losses = []

        for epoch in range(num_epochs):
            # Run validation batch
            for i, (images, annotations) in enumerate(self._validation_data):
                if i in indices_to_skip:
                    continue

                # Move all images and annotation values to the correct device
                images = tuple([image.to(self._device) for image in images])
                annotations = [{key: value.to(self._device) for key, value in target.items()} for target in annotations]

                # Do forward propagation and get the loss values for the image
                # NOTE: validation_loss is a dictionary with different loss values. See class description for more info.
                validation_loss = self._frcnn(images, annotations)
                validation_loss = [loss for loss in validation_loss.values()]
                validation_loss = sum(validation_loss)

                validation_epoch_loss += validation_loss

                if not self._quiet or self._debug:
                    print(f"Batch: {i}/{len(self._train_data)} | Batch Loss: {validation_loss}\r", end="")

            validation_losses.append(validation_epoch_loss)

            if not self._quiet:
                print(f"\n\n############# Epoch: {epoch} Complete #############")
                print(f"\t- Epoch Loss: {validation_epoch_loss}\n\n")

        self._validation_metrics['LOSS'] = validation_losses

    def model_test(self, test_data: torch.utils.data.DataLoader, quiet: bool = True):
        """
        Run Testing on the given model for this worker.

        Args:
            - test_data: Loader object for all testing data.
            - quiet: Whether to silence print statements.
        """
        # Set the model to evaluation mode
        self._frcnn.eval()

        testing_losses = []

        with torch.no_grad():

            for test_batch, (images, annotations) in enumerate(test_data):
                # try:
                images = tuple([image.to(self._device) for image in images])
                annotations = [{key: value.to(self._device) for key, value in target.items()} for target in annotations]
                # except:
                #     print(f"Failed on Batch: {test_batch}\n Image: {images}\n Annotations: {annotations}")

                # try:
                test_loss = self._frcnn(images, annotations)
                print(test_loss)
                break
                test_loss = [loss for loss in test_loss.values()]
                test_loss = sum(test_loss)

                testing_losses.append(test_loss.item())

                if not quiet:
                    print(f"Batch: {test_batch}/{len(test_data)} | Batch Loss: {test_loss}\r", end="")
                # except:
                #     print(f"Failed on Batch: {test_batch}\n Image: {images}\n Annotations: {annotations}")

        self._test_metrics['LOSS'] = testing_losses