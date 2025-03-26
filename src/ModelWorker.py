import torch

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