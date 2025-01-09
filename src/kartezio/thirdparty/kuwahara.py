# First try at manually rewriting the Kuwahara function from Luca Balbi which was written
# originally in MatLab
import cv2
import numpy as np


def kuwahara_filter(original, winsize):
    """
    Kuwahara filters an image using the Kuwahara filter

    Inputs:
    original      -->    image to be filtered
    windowSize    -->    size of the filter window: legal values are
                                                    5, 9, 13, ... = (4*k+1)
    This function is optimised using vectorialisation, convolution and the fact
    that, for every subregion
        variance = (mean of squares) - (square of mean).
    A nested-for loop approach is still used in the final part as it is more
    readable, a commented-out, fully vectorialised version is provided as well.

    Example
    filtered = Kuwahara(original,5);

    Filter description:
    The Kuwahara filter works on a window divided into 4 overlapping
    subwindows (for a 5x5 pixels example, see below). In each subwindow, the mean and
    variance are computed. The output value (located at the center of the
    window) is set to the mean of the subwindow with the smallest variance.

        ( a  a  ab   b  b)
        ( a  a  ab   b  b)
        (ac ac abcd bd bd)
        ( c  c  cd   d  d)
        ( c  c  cd   d  d)

    References:
    http://www.ph.tn.tudelft.nl/DIPlib/docs/FIP.pdf
    http://www.incx.nec.co.jp/imap-vision/library/wouter/kuwahara.html


    Copyright Luca Balbi, 2007
    Original license is contained in a block comment at the bottom of this file.

    Translated from Matlab into Python by Andrew Dussault, 2015
    from https://github.com/adussault/python-kuwahara
    """

    image = original.astype(np.float32)

    # Build subwindows
    height, width = image.shape[:2]

    # Padding the image to handle borders
    half_winsize = (winsize - 1) // 2

    tmpAvgKerRow = np.hstack(
        (np.ones((1, half_winsize + 1)), np.zeros((1, half_winsize)))
    )
    tmpPadder = np.zeros((1, winsize))
    tmpavgker = np.tile(tmpAvgKerRow, (half_winsize + 1, 1))
    tmpavgker = np.vstack((tmpavgker, np.tile(tmpPadder, (half_winsize, 1))))
    tmpavgker = tmpavgker / np.sum(tmpavgker)

    # tmpavgker is a 'north-west' subwindow (marked as 'a' above)
    # we build a vector of convolution kernels for computing average and
    # variance
    avgker = np.empty((4, winsize, winsize))  # make an empty vector of arrays
    avgker[0] = tmpavgker  # North-west (a)
    avgker[1] = np.fliplr(tmpavgker)  # North-east (b)
    avgker[2] = np.flipud(tmpavgker)  # South-west (c)
    avgker[3] = np.fliplr(avgker[2])  # South-east (d)

    # Create a pixel-by-pixel square of the image
    squaredImg = image**2

    # preallocate these arrays to make it apparently %15 faster
    avgs = np.zeros([4, height, width])
    stddevs = avgs.copy()

    # Calculation of averages and variances on subwindows
    for k in range(4):
        avgs[k] = cv2.filter2D(
            src=image, ddepth=-1, kernel=avgker[k]
        )  # convolve2d(image, avgker[k],mode='same') 	    # mean on subwindow
        stddevs[k] = cv2.filter2D(
            src=squaredImg, ddepth=-1, kernel=avgker[k]
        )  # convolve2d(squaredImg, avgker[k],mode='same')  # mean of squares on subwindow
        stddevs[k] = (stddevs[k] - avgs[k]) ** 2  # variance on subwindow

    # Choice of index with minimum variance
    indices = np.argmin(
        stddevs, 0
    )  # returns index of subwindow with smallest variance

    # Building the filtered image (with nested for loops)
    filtered = np.zeros(original.shape)
    for row in range(original.shape[0]):
        for col in range(original.shape[1]):
            filtered[row, col] = avgs[indices[row, col], row, col]

    # filtered=filtered.astype(np.uint8)
    return filtered.astype(np.uint8)


"""
ORIGINAL LICENSE OF MATLAB CODE:
Copyright (c) 2007, Luca Balbi
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the distribution
    * Neither the name of the Esaote S.p.A. nor the names
      of its contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
