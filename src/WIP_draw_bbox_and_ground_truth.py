import matplotlib.pyplot as plt
import matplotlib.patches as patches

def model_test(frcnn, device, test_data: torch.utils.data.DataLoader, threshold: float = 0.8):
        """
        Run Testing on the given model for this worker.

        Args:
            - test_data: Loader object for all testing data.
            - threshold: Minimum bounding box score to consider in our metric calculations
        """
        import numpy as np
        # Set the model to evaluation mode
        frcnn.eval()
        test_results = {}

        with torch.no_grad():
            for test_batch, (images, annotations) in enumerate(test_data):
                images = tuple([image.to(device) for image in images])
                annotations = [{key: value.to(device) for key, value in target.items()} for target in annotations]

                pred = frcnn(images, annotations)
                # print(pred)

                scores = pred[0]['scores'].cpu().detach().numpy()
                boxes = pred[0]['boxes'].cpu().detach().numpy()
                labels = pred[0]['labels'].cpu().detach().numpy()
                # print(scores)
                # Extract indices that have scores exceeding minimum threshold.
                # These are predictions that are closest to the ground truth
                # print(scores)
                # high_confidence_indices = np.where(np.any(scores > 0))
                valid_boxes = []
                for i in range(len(scores)):
                     if scores[i] > threshold:
                          valid_boxes.append(boxes[i])
                # print(valid_scores)
                # valid_scores = scores[high_confidence_indices]
                # valid_boxes = boxes[valid_scores]
                # print(valid_boxes)
                # valid_labels = labels[high_confidence_indices]
                # print(high_confidence_indices)
                # Extract bounding boxes, scores, and labels from valid predictions.
                test_results[test_batch] = {
                    'boxes': [],
                    'scores': [],
                    'labels': []
                }
                for image in images[test_batch]:
                    print(image)
                    plt.imshow(image.cpu().detach().numpy())
                # print(valid_boxes)
                # plt.imshow(images[test_batch].cpu().permute(1, 2, 0))
                # for prediction in range(len(valid_boxes)):
                #     width = valid_boxes[prediction][2] - valid_boxes[prediction][0]
                #     height = valid_boxes[prediction][3] - valid_boxes[prediction][1]

                #     ax = plt.gca()
                #     rect = patches.Rectangle([valid_boxes[prediction][0], valid_boxes[prediction][1]], width, height, edgecolor='r', facecolor='none')
                #     ax.add_patch(rect)
                #     # plt.show()
                # #     test_results[test_batch]['boxes'].append(valid_boxes[prediction])
                # #     test_results[test_batch]['scores'].append(valid_scores[prediction])
                # #     test_results[test_batch]['labels'].append(valid_labels[prediction])
                # plt.show()
                break

        # print(test_results)