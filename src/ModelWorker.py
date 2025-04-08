from copy import deepcopy
import gc
import torch
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as patches


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
                 optimizer: torch.optim.Optimizer,
                 frcnn: torch.nn.Module,
                 quiet: bool = True,
                 debug: bool = False):

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
        self._train_metrics = {'loss': None}
        self._validation_metrics = {'loss': None}
        self._test_results = {}

    def train(self, train_dataloader: torch.utils.data.DataLoader, num_epochs: int, indices_to_skip: list = []):
        """
        Run Training step.

        Args:
            - train_dataloader: The training data loader object.
            - num_epochs: Number of training iterations to run.
            - skip_index: Data instances to skip during training
                - Some data instances are missing their annotations in the Kaggle dataset.  We will skip these and notify the
                  creator.
                - Known Indices: [3736]
        """

        self._frcnn.train()
        training_epoch_losses = {'total_loss': [], 'loss_objectness': [], 'loss_rpn_box_reg': [], 'loss_box_reg': [], 'loss_classifier': []}

        for epoch in range(num_epochs):
            for i, (images, annotations) in enumerate(train_dataloader):
                train_batch_losses = {'total_loss': 0, 'loss_objectness': 0, 'loss_rpn_box_reg': 0, 'loss_box_reg': 0, 'loss_classifier': 0}
                if i in indices_to_skip:
                    # continue
                    break
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
                    train_loss_dict = train_loss

                    # Calculate total batch loss across all categories for backpropagation
                    train_loss = [loss for loss in train_loss.values()]
                    train_loss = sum(train_loss)

                    # Convert the losses to regular python values for metric tracking
                    train_loss_dict = {key: value.cpu().detach().numpy().tolist() for key, value in train_loss_dict.items()}

                    train_batch_losses['total_loss'] += train_loss
                    train_batch_losses['loss_objectness'] += train_loss_dict['loss_objectness']
                    train_batch_losses['loss_rpn_box_reg'] += train_loss_dict['loss_rpn_box_reg']
                    train_batch_losses['loss_box_reg'] += train_loss_dict['loss_box_reg']
                    train_batch_losses['loss_classifier'] += train_loss_dict['loss_classifier']

                    # Backpropagation
                    self._optimizer.zero_grad()
                    train_loss.backward()
                    self._optimizer.step()

                    if not self._quiet or self._debug:
                        print(f"Batch: {i}/{len(train_dataloader)} | Batch Loss: {train_batch_losses['total_loss']}\r", end="")

                except Exception as e:
                    print(e)
                    print(f"Failed on Batch {i}")
                    print(f"Image: {images}")
                    print(f"Annotations: {annotations}")
                    sys.exit(-1)

            training_epoch_losses['total_loss'].append(train_batch_losses['total_loss'])
            training_epoch_losses['loss_objectness'].append(train_batch_losses['loss_objectness'])
            training_epoch_losses['loss_rpn_box_reg'].append(train_batch_losses['loss_rpn_box_reg'])
            training_epoch_losses['loss_box_reg'].append(train_batch_losses['loss_box_reg'])
            training_epoch_losses['loss_classifier'].append(train_batch_losses['loss_classifier'])

            if not self._quiet:
                print(f"\n\n############# Epoch: {epoch} Complete #############")
                print(f"Total Batch Loss: {training_epoch_losses['total_loss'][epoch]}")
                print(f"Loss Objectness: {training_epoch_losses['loss_objectness']}")
                print(f"RPN Region Proposal Losses: {training_epoch_losses['loss_rpn_box_reg'][epoch]}")
                print(f"Classifier Loss: {training_epoch_losses['loss_classifier'][epoch]}")
                print(f"Bounding Box Region Loss: {training_epoch_losses['loss_box_reg'][epoch]}")

            self._train_metrics['loss'] = training_epoch_losses

            # Free all memory from the current batch
            images = None
            annotations = None
            train_loss = None
            train_epoch_loss = None
            del images
            del annotations
            del train_loss
            del train_epoch_loss
            gc.collect()
            torch.cuda.empty_cache()

        self._train_metrics['LOSS'] = training_epoch_losses

        # Clean up memory from training
        training_epoch_losses = None
        self._optimizer = None
        del training_epoch_losses
        del self._optimizer
        gc.collect()
        torch.cuda.empty_cache()

    def validation(self, validation_dataloader: torch.utils.data.DataLoader, num_epochs: int, indices_to_skip: list = []):
        """
        Run Validation step.

        Args:
            - validation_dataloader: Validation data place within Pytorch dataloader object
            - num_epochs: Number of epochs to run validation.
        """

        self._frcnn.train()
        validation_epoch_losses = {'total_loss': [], 'loss_objectness': [], 'loss_rpn_box_reg': [], 'loss_box_reg': [], 'loss_classifier': []}
        with torch.no_grad():
            for epoch in range(num_epochs):
                validation_batch_losses = {'total_loss': 0, 'loss_objectness': 0, 'loss_rpn_box_reg': 0, 'loss_box_reg': 0, 'loss_classifier': 0}
                # Run validation batch
                for i, (images, annotations) in enumerate(validation_dataloader):
                    if i in indices_to_skip:
                        # continue
                        break
                    # Move all images and annotation values to the correct device
                    images = tuple([image.to(self._device) for image in images])
                    annotations = [{key: value.to(self._device) for key, value in target.items()} for target in annotations]

                    # Do forward propagation and get the loss values for the image
                    # NOTE: validation_loss is a dictionary with different loss values. See class description for more info.
                    validation_loss = self._frcnn(images, annotations)
                    validation_loss_dict = validation_loss

                    validation_loss = [loss for loss in validation_loss.values()]
                    validation_loss = sum(validation_loss)
                    validation_loss_dict = {key: value.cpu().detach().numpy().tolist() for key, value in validation_loss_dict.items()}
                    validation_batch_losses['total_loss'] += validation_loss
                    validation_batch_losses['loss_objectness'] += validation_loss_dict['loss_objectness']
                    validation_batch_losses['loss_rpn_box_reg'] += validation_loss_dict['loss_rpn_box_reg']
                    validation_batch_losses['loss_box_reg'] += validation_loss_dict['loss_box_reg']
                    validation_batch_losses['loss_classifier'] += validation_loss_dict['loss_classifier']

                    if not self._quiet or self._debug:
                        print(f"Batch: {i}/{len(validation_dataloader)} | Batch Loss: {validation_loss}\r", end="")

                    # Clean up from current batch
                    validation_loss = None
                    images = None
                    annotations = None
                    del validation_loss
                    del images
                    del annotations
                    gc.collect()
                    torch.cuda.empty_cache()

                validation_epoch_losses['loss_objectness'].append(validation_loss['loss_objectness'])
                validation_epoch_losses['loss_rpn_box_reg'].append(validation_loss['loss_rpn_box_reg'])
                validation_epoch_losses['loss_box_reg'].append(validation_loss['loss_box_reg'])
                validation_epoch_losses['loss_classifier'].append(validation_loss['loss_classifier'])

                if not self._quiet:
                    print(f"\n\n############# Epoch: {epoch} Complete #############")
                    print(f"\t- Epoch Loss: {validation_epoch_losses}\n\n")
                    print(f"Total Batch Loss: {validation_epoch_losses['total_loss'][epoch]}")
                    print(f"Loss Objectness: {validation_epoch_losses['loss_objectness']}")
                    print(f"RPN Region Proposal Losses: {validation_epoch_losses['loss_rpn_box_reg'][epoch]}")
                    print(f"Classifier Loss: {validation_epoch_losses['loss_classifier'][epoch]}")
                    print(f"Bounding Box Region Loss: {validation_epoch_losses['loss_box_reg'][epoch]}")

            self._validation_metrics['loss'] = validation_epoch_losses

            # Clean up final validation variables for memory conservation
            del validation_losses
            gc.collect()
            torch.cuda.empty_cache()

    def model_test(self, test_data: torch.utils.data.DataLoader, threshold: float = 0.8):
        """
        Run Testing on the given model for this worker.

        Args:
            - test_data: Loader object for all testing data.
            - threshold: Minimum bounding box score to consider in our metric calculations
        """
        import numpy as np
        # Set the model to evaluation mode
        self._frcnn.eval()

        with torch.no_grad():
            for test_batch, (images, annotations) in enumerate(test_data):
                images = tuple([image.to(self._device) for image in images])
                annotations = [{key: value.to(self._device) for key, value in target.items()} for target in annotations]

                pred = self._frcnn(images, annotations)

                valid_predictions = {}
                # Convert all results in the batch to numpy arrays
                # Extract indices that have scores exceeding minimum threshold.
                # These are predictions that are closest to the ground truth
                for prediction in range(len(pred)):
                    scores = pred[prediction]['scores'].cpu().detach().numpy()
                    bboxes = pred[prediction]['boxes'].cpu().detach().numpy()
                    valid_predictions[prediction] = {'scores': [], 'bboxes': []}
                    for i in range(len(scores)):
                        if scores[i] > threshold:
                            valid_predictions[prediction]['scores'].append(scores[i])
                            valid_predictions[prediction]['bboxes'].append(bboxes[i])
                print(valid_predictions)
                for image in images:
                    plt.imshow(image.cpu().permute(1, 2, 0))
                    for bbox in valid_predictions[0]['bboxes']:
                        try:
                            width = bbox[2] - bbox[0]
                            height = bbox[3] - bbox[1]

                            ax = plt.gca()
                            rect = patches.Rectangle([bbox[0], bbox[1]], width, height, edgecolor='r', facecolor='none')
                            ax.add_patch(rect)
                        except Exception as e:
                            print(e)
                    plt.show()
                break

        # return 0
