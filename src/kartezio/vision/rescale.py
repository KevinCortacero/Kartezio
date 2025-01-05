import cv2


def pyrdown(image, n=1):
    """
    Downsample an image n times using cv2.pyrDown.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    n : int, optional
        Number of times to downsample (default is 1).

    Returns
    -------
    np.ndarray
        The downsampled image.
    """
    for _ in range(n):
        image = cv2.pyrDown(image)
    return image


def pyrup(image, n=1):
    """
    Upsample an image n times using cv2.pyrUp.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    n : int, optional
        Number of times to upsample (default is 1).

    Returns
    -------
    np.ndarray
        The upsampled image.
    """
    for _ in range(n):
        image = cv2.pyrUp(image)
    return image
