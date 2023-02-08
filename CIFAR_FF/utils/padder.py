import numpy as np

def pad_to_size(images, to_size=(32, 32)):
    """ A general zero padding function into given size if needed.
    - Also adds a missing dimension for black and white images.
    - Don't do anything if the image dimensions are greater than the given 
    """
    # Add the missing dimension if image has only luminocity channel such as MNIST
    if len(images.shape) == 3:
        images = np.expand_dims(images, axis=3)
           
    _, h, w, _ = images.shape

    # Calculate the needed padding
    diff_h = (to_size[0]-h)
    diff_w = (to_size[1]-w)
    if diff_h >= 0 and diff_w >= 0:
        return images  # don't do anything

    # Calculate axis-wise padding and add any surplus to the bottom-right corner
    pad_h1 = diff_h // 2
    pad_h2 = diff_h // 2 + diff_h % 2
    pad_h = (pad_h1, pad_h2)
    
    pad_w1 = diff_w // 2
    pad_w2 = diff_w // 2 + diff_w % 2  
    pad_w = (pad_w1, pad_w2)
        
    padded_images = np.pad(images, ((0,0), pad_h, pad_w, (0,0)), mode='constant')

    return padded_images