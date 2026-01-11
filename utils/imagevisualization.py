import os
import cv2
import numpy as np
import utils.colors


LINE_THICKNESS = 2
TEXT_THICKNESS = 1
TEXT_SIZE = 0.7
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_LINE_TYPE = cv2.LINE_AA


def drawCV2BoundingBox(image, tl, br, color, thickness=LINE_THICKNESS):
    """
    Draw a bounding box using CV2.
    """
    cv2.rectangle(image, tl, br, color, thickness)


def drawCV2Text(image, text, bl, color, thickness=TEXT_THICKNESS, 
                size=TEXT_SIZE):
    """
    Add text using CV2.
    """
    cv2.putText(image, text, bl, cv2.FONT_HERSHEY_SIMPLEX, size, color, 
                thickness, lineType=cv2.LINE_AA)


def drawCV2BBWithText(image, bbSerialized, text, color, 
                      bbThickness=LINE_THICKNESS, 
                      textThickness=TEXT_THICKNESS, 
                      textSize=TEXT_SIZE):
    """
    Draw bounding box with optional text on top.
    The bounding box is Serialized: xTL, yTL, xBR, yBR.
    """
    cv2.rectangle(
        image,
        (int(bbSerialized[0]), int(bbSerialized[1])),
        (int(bbSerialized[2]), int(bbSerialized[3])),
        color,
        bbThickness
    )
    
    if text:
        cv2.putText(
            image,
            text,
            (int(bbSerialized[0]), int(bbSerialized[1] - 5)),
            TEXT_FONT,
            textSize,
            color,
            textThickness,
            lineType=TEXT_LINE_TYPE
        )


def visualizeImage(image, target, classes, classescolors=None):
    """
    Visualize the image with bounding box.
    Random colors used if class colors are not given.
    """
    if classescolors:
        colorpalette = {str(v): classescolors[k] for k, v in classes.items()}
    else:
        colorpalette = {str(v): utils.colors.getRandomColor(2) 
                       for k, v in classes.items()}

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    for box_num in range(len(target['boxes'])):
        box = target['boxes'][box_num]
        label = str(target['labels'][box_num].numpy())
        color = colorpalette[label]
        drawCV2BBWithText(image, box, label, color)

    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# TODO: Use class and random colors
def visualizeOneBatchImages(batch, max_images=5):
    """
    Visualize one batch of images with bounding box from a DEVICE.
    """
    images, targets, names = batch
    loopMax = max_images if len(images) > max_images else len(images)
    
    for ii in range(loopMax):
        image, target = images[ii], targets[ii]
        boxes = target['boxes'].cpu().numpy().astype(np.int32)
        labels = target['labels'].cpu().numpy()
        sample = image.permute(1, 2, 0).cpu().numpy()
        
        for box_num, box in enumerate(boxes):
            drawCV2BBWithText(sample, box, str(labels[box_num]), (255, 0, 0))
        
        cv2.imshow('Transformed image', sample)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def visualizeOneBatchImagesFromLoader(train_loader, device, max_images=5):
    """
    Visualize one batch of images with bounding box from train_loader.
    """
    if len(train_loader.dataset) > 0:
        images, targets, _ = next(iter(train_loader))
        loopMax = max_images if len(images) > max_images else len(images)
        
        for ii in range(loopMax):
            image, target = images[ii], targets[ii]
            image = image.to(device)
            target = {k: v.to(device) for k, v in target.items()}
            boxes = target['boxes'].cpu().numpy().astype(np.int32)
            labels = target['labels'].cpu().numpy()
            sample = image.permute(1, 2, 0).cpu().numpy()
            
            for box_num, box in enumerate(boxes):
                drawCV2BBWithText(
                    sample, box, str(labels[box_num]), (255, 255, 255)
                )
            
            cv2.imshow('Transformed image', sample)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def saveOneBatchImages(train_loader, savepath):
    """
    Save one batch of images from loader.
    """
    os.makedirs(savepath, exist_ok=True)
    
    if len(train_loader.dataset) > 0:
        images, targets, imagenames = next(iter(train_loader))
        
        for ii in range(len(images)):
            image, image_name = images[ii], imagenames[ii]
            image_save_path = os.path.join(savepath, image_name)
            numpy_image = np.transpose(image.numpy(), (1, 2, 0))
            cv2.imwrite(image_save_path, numpy_image * 255)