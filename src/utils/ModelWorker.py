import gc
import torch
import sys
import matplotlib.pyplot as plt
from torchmetrics.functional.detection import intersection_over_union
from torchmetrics.detection import MeanAveragePrecision


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
        self._train_metrics = {'LOSS': None, 'num_epochs': None}
        self._validation_metrics = {'LOSS': None, 'num_epochs': None}

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
        self._train_metrics['num_epochs'] = num_epochs
        self._frcnn.train()
        training_epoch_losses = {'epoch_loss': [], 'loss_objectness': [], 'loss_rpn_box_reg': [], 'loss_box_reg': [], 'loss_classifier': []}

        for epoch in range(num_epochs):
            train_batch_losses = {'epoch_loss': 0, 'loss_objectness': 0, 'loss_rpn_box_reg': 0, 'loss_box_reg': 0, 'loss_classifier': 0}
            for i, (images, annotations) in enumerate(train_dataloader):
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
                    if isinstance(train_loss, tuple):
                        train_loss = train_loss[0]

                    train_loss_dict = train_loss

                    # Calculate total batch loss across all categories for backpropagation
                    train_loss = [loss for loss in train_loss.values()]
                    train_loss = sum(train_loss)

                    # Convert the losses to regular python values for metric tracking
                    train_loss_dict = {key: value.cpu().detach().numpy().tolist() for key, value in train_loss_dict.items()}

                    train_batch_losses['epoch_loss'] += float(train_loss.cpu().detach())
                    train_batch_losses['loss_objectness'] += train_loss_dict['loss_objectness']
                    train_batch_losses['loss_rpn_box_reg'] += train_loss_dict['loss_rpn_box_reg']
                    train_batch_losses['loss_box_reg'] += train_loss_dict['loss_box_reg']
                    train_batch_losses['loss_classifier'] += train_loss_dict['loss_classifier']

                    # Backpropagation
                    self._optimizer.zero_grad()
                    train_loss.backward()
                    self._optimizer.step()

                    if not self._quiet or self._debug:
                        print(f"Batch: {i}/{len(train_dataloader)} | Batch Loss: {train_loss}\r", end="")

                except Exception as e:
                    print(e)
                    print(f"Failed on Batch {i}")
                    print(f"Image: {images}")
                    print(f"Annotations: {annotations}")
                    sys.exit(-1)

            training_epoch_losses['epoch_loss'].append(train_batch_losses['epoch_loss'])
            training_epoch_losses['loss_objectness'].append(float(train_batch_losses['loss_objectness']))
            training_epoch_losses['loss_rpn_box_reg'].append(float(train_batch_losses['loss_rpn_box_reg']))
            training_epoch_losses['loss_box_reg'].append(float(train_batch_losses['loss_box_reg']))
            training_epoch_losses['loss_classifier'].append(float(train_batch_losses['loss_classifier']))

            if not self._quiet:
                print(f"\n\n############# Epoch: {epoch} Complete #############")
                print(f"Total Epoch Combined Loss: {training_epoch_losses['epoch_loss'][epoch]}")
                print(f"Loss Objectness: {training_epoch_losses['loss_objectness'][epoch]}")
                print(f"RPN Region Proposal Losses: {training_epoch_losses['loss_rpn_box_reg'][epoch]}")
                print(f"Classifier Loss: {training_epoch_losses['loss_classifier'][epoch]}")
                print(f"Bounding Box Region Loss: {training_epoch_losses['loss_box_reg'][epoch]}")
            print("\n")

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
        self._validation_metrics['num_epochs'] = num_epochs
        self._frcnn.train()
        validation_epoch_losses = {'epoch_loss': [], 'loss_objectness': [], 'loss_rpn_box_reg': [], 'loss_box_reg': [], 'loss_classifier': []}
        with torch.no_grad():
            for epoch in range(num_epochs):
                validation_batch_losses = {'epoch_loss': 0, 'loss_objectness': 0, 'loss_rpn_box_reg': 0, 'loss_box_reg': 0, 'loss_classifier': 0}
                # Run validation batch
                for i, (images, annotations) in enumerate(validation_dataloader):
                    if i in indices_to_skip:
                        continue
                    # Move all images and annotation values to the correct device
                    images = tuple([image.to(self._device) for image in images])
                    annotations = [{key: value.to(self._device) for key, value in target.items()} for target in annotations]

                    # Do forward propagation and get the loss values for the image
                    # NOTE: validation_loss is a dictionary with different loss values. See class description for more info.
                    validation_loss = self._frcnn(images, annotations)
                    if isinstance(validation_loss, tuple):
                        validation_loss = validation_loss[0]

                    validation_loss_dict = validation_loss

                    validation_loss = [loss for loss in validation_loss.values()]
                    validation_loss = sum(validation_loss)
                    validation_loss_dict = {key: value.cpu().detach().numpy().tolist() for key, value in validation_loss_dict.items()}
                    validation_batch_losses['epoch_loss'] += validation_loss.cpu().detach()
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

                validation_epoch_losses['epoch_loss'].append(validation_batch_losses['epoch_loss'])
                validation_epoch_losses['loss_objectness'].append(float(validation_batch_losses['loss_objectness']))
                validation_epoch_losses['loss_rpn_box_reg'].append(float(validation_batch_losses['loss_rpn_box_reg']))
                validation_epoch_losses['loss_box_reg'].append(float(validation_batch_losses['loss_box_reg']))
                validation_epoch_losses['loss_classifier'].append(float(validation_batch_losses['loss_classifier']))

                if not self._quiet:
                    print(f"\n\n############# Epoch: {epoch} Complete #############")
                    print(f"\tTotal Epoch Loss: {validation_epoch_losses['epoch_loss'][epoch]}")
                    print(f"\tLoss Objectness: {validation_epoch_losses['loss_objectness']}")
                    print(f"\tRPN Region Proposal Losses: {validation_epoch_losses['loss_rpn_box_reg'][epoch]}")
                    print(f"\tClassifier Loss: {validation_epoch_losses['loss_classifier'][epoch]}")
                    print(f"\tBounding Box Region Loss: {validation_epoch_losses['loss_box_reg'][epoch]}")

            self._validation_metrics['LOSS'] = validation_epoch_losses

            # Clean up final validation variables for memory conservation
            del validation_batch_losses
            gc.collect()
            torch.cuda.empty_cache()

    def model_test(self, test_data: torch.utils.data.DataLoader, threshold: float = 0.8):
        """
        Run Testing on the given model for this worker.

        Args:
            - test_data: Loader object for all testing data.
            - threshold: Minimum IOU score to consider in our metric calculations.
        """
        performance = {"Image": [], "Ground Truth": [], "Prediction": []}

        maP = MeanAveragePrecision()
        # Set the model to evaluation mode
        self._frcnn.eval()
        with torch.no_grad():
            for test_batch, (images, annotations) in enumerate(test_data):
                images = tuple([image.to(self._device) for image in images])
                annotations = [{key: value.to(self._device) for key, value in target.items()} for target in annotations]

                test_prediction = self._frcnn(images, annotations) # Make a prediction on the current image.
                maP.update(test_prediction, annotations)
                test_boxes = test_prediction[0]['boxes'].cpu().detach()

                ground_truth = []
                # Find the Intersection over Union for each bounding box compared to the ground truth annotation
                for gt in annotations[0]['boxes']:
                    ground_truth.append(torch.unsqueeze(gt.cpu().detach(), 0)[0])

                ground_truth = torch.stack(ground_truth)

                performance["Ground Truth"].append(ground_truth)
                performance["Prediction"].append(test_boxes)
                performance["Image"].append([image.cpu().detach() for image in images])

                if not self._quiet:
                    print(f"Test Batch: {test_batch}\r", end="")

        return performance, maP

    def plot_losses(self, plot_train: bool = True):
        fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(26, 5))
        if plot_train:
            for i, (key, value) in enumerate(self._train_metrics['LOSS'].items()):
                ax[i].plot(range(0, self._train_metrics['num_epochs']), value)
                ax[i].set_title(f"Training Epoch {key}")
                ax[i].set_xlabel("Epochs")
                ax[i].set_ylabel(key)
        else:
            for i, (key, value) in enumerate(self._validation_metrics['LOSS'].items()):
                ax[i].plot(range(0, self._validation_metrics['num_epochs']), value)
                ax[i].set_title(f"Validation Epoch {key}")
                ax[i].set_xlabel("Epochs")
                ax[i].set_ylabel(key)

    def get_test_metrics(self, performance: dict, threshold: float):
        """
        Calculates the Intersection over Union, true positives, false positives, and false negatives.
        Then uses those to find the Precision and Recall for each ground truth element.

        According to this link: https://www.comet.com/site/blog/compare-object-detection-models-from-torchvision/,
            - True Positives are predicted bounding boxes whose IOU is greater than the threshold.
            - False Positives are predicted bounding boxes whose IOU is less than the threshold.
            - False Negatives are ground truth bounding boxes with no predictions made on them.

        Args:
            - performance: Dictionary containing the test result bounding boxes.
            - threshold: The Intersection over Union threshold.

        Returns:
            - performance: Updated dictionary containing the IOU, TP, FP, and FN values.
        """

        performance['True Positives'] = 0
        performance['False Positives'] = 0
        performance['False Negatives'] = 0
        performance['Precisions'] = []
        performance['Recalls'] = []
        performance['Best IOU'] = []
        for i, truth_boxes in enumerate(performance['Ground Truth']):
            for box in truth_boxes:
                if len(performance['Prediction'][i]) == 0:
                    performance['False Negatives'] += 1
                else:
                    best_iou = torch.max(intersection_over_union(performance['Prediction'][i], box.unsqueeze(0)))
                    if best_iou >= threshold:
                        performance['True Positives'] += 1
                    elif best_iou < threshold:
                        performance['False Positives'] += 1

                    performance['Best IOU'].append(best_iou)

                tp, fp, fn = performance['True Positives'], performance['False Positives'], performance['False Negatives']

                if tp > 0 or fp > 0:
                    precision = tp / (tp + fp)
                else:
                    precision = 0

                if tp > 0 or fn > 0:
                    recall = tp / (tp + fn)
                else:
                    recall = 0

                performance['Precisions'].append(precision)
                performance['Recalls'].append(recall)

        return performance

    def plot_PR_curve(self, performance: dict):
        """
        Plots the Precision-Recall curve.  performance dictionary must have True Positives, False Positives,
        and False Negatives keys from the get_test_metrics() method.

        Args:
            - performance: Dictionary containing most up-to-date metrics.
        """
        plt.plot(performance['Recalls'], performance['Precisions'])
        plt.xlabel("Recall")
        plt.ylabel("Precision")