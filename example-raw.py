#!/usr/bin/env python

import gzip
import rawpy
import numpy as np
from scipy.signal import convolve2d

# NB: I have to run this stuff for pylab
# to work correctly on OSX... not sure
# if you might need to tweak the "TkAgg"
# option to make it work on your system.
import matplotlib
matplotlib.use('TkAgg')
import pylab as pl
pl.ion()


'''
Note!!! This just copied from example-jpg.py and is almost
exactly the same, but simpler since there are no color
channels, etc, in the RAW images.
Mostly this just serves as an example of how to load
RAW images into numpy format.

you'll need to install the rawpy package from github directly:
    pip install --upgrade git+git://github.com/letmaik/rawpy.git#egg=rawpy

So go through example-jpg.py
first.
'''
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Example for processing RAW images")

    # protip: nargs="+" means accept one or more arguments from the command
    # line; they will be given to us as a list regardless.
    parser.add_argument("input_image", nargs="+", help="input jpeg file(s)")

    args = parser.parse_args()

    # open up all the input images, keeping just their data as a
    # list of numpy arrays
    print "Opening %d files..." % len(args.input_image)
    imgs = []
    for fname in args.input_image:
        # check if the file is gzip-compressed and choose
        # the appropriate method for opening it.
        if fname.endswith(".gz"):
            f = gzip.open(fname)
        else:
            f = open(fname)

        # read a RAW file object with the rawpy library:
        raw = rawpy.imread(f)

        # raw.raw_image is a (w,h)-shaped numpy array with
        # the RAW 10-bit pixel values from 0-1023.
        imgs.append(raw.raw_image)

    # Ok, right now each image array has the shape (w,h);
    # so an single integer at each pixel. and we have a _list_
    # of these arrays. as before we're going to turn the whole
    # sequence of images into a giant multidimensional array.
    imgs = np.array(imgs)
    # now we have an array with shape (N, w, h), where
    # N is the number of images passed in on the command line.
    
    # let's draw a preview of the first image:
    print "Drawing image"
    pl.figure()
    pl.imshow(imgs[0])
    pl.colorbar()
    pl.title("img0 RAW")
    pl.draw() # force immmediate graphics update when running in commandline mode

    # let's histogram the pixel values in the first image
    pl.figure()
    xmax = int(np.ceil(imgs[0].max()))
    pl.hist(imgs[0].flat, histtype='step', bins=xmax, range=(0,xmax))
    pl.title('img0 RAW statistics')
    pl.xlabel('pix val')
    pl.yscale('log'); pl.ylim(ymin=1e-1)
    pl.draw()

    # as before calculate some per-pixel stats across the images.
    pix_avg = imgs.mean(axis=0)
    pix_var = imgs.var(axis=0)
    pix_max = imgs.max(axis=0)

    # let's histogram these pixel results; this shows us the statistical
    # characterization of each individual pixel channel over the multiple
    # example images.
    pl.figure()
    xmax = int(np.ceil(pix_max.max()))
    pl.hist(pix_avg.flat, histtype='step', bins=xmax, range=(0,xmax), label='avg pix value')
    pl.hist(pix_max.flat, histtype='step', bins=xmax, range=(0,xmax), label='max pix value')
    pl.yscale('log'); pl.ylim(ymin=1e-1)
    pl.legend(loc='best')
    pl.title('per-pixel statistics')
    pl.xlabel('pix val')
    pl.draw()

    pl.figure()
    pl.hist(pix_var.flat, histtype='step', bins=200, label='var. pix value')
    pl.legend(loc='best')
    pl.yscale('log'); pl.ylim(ymin=1e-1)
    pl.title('per-pixel statistics')
    pl.xlabel('pix variance')
    pl.draw()

    # one idea to look for interesting "hits" across multiple images is to
    # just look at an image of the maximum pixel value over several images.
    # if a cosmic hit occurs and leaves a bright track in any one of the images,
    # it should show up in this single composite as well:
    pl.figure()
    pl.imshow(pix_max)
    pl.colorbar()
    pl.title("pixel-max composite")
    pl.draw()


    # finally, let's look at calculating something like avg3; it is the average
    # of each pixel's surrounding 3x3 box of pixels. this can be expressed and
    # efficiently calculated as a 2d matrix convolution with the following kernel:

    # start with 3x3 array of ones (i.e. [[1 1 1] [1 1 1] [1 1 1]])
    k3 = np.ones((3,3), dtype=float)
    # punch out the middle one (since we don't include the pixel itself in the avg)
    k3[1,1] = 0
    # normalize the kernel to unity:
    k3 /= 8.

    # now let's look at the resulting convolution with the L channel
    # of the first input image. note that the option 'same' forces the
    # output of the convolution to have the same dimensions as the original
    # image; this means that edge effects will be present since the "average"
    # at the edges isn't correctly handleded (numpy just treats the missing
    # pixels at the edge to have value 0, so they bring down the average).
    print "calculating local pixel averages..."
    avg3 = convolve2d(imgs[0], k3, mode='same')

    # while we're at it, we could generically caluclate the average for any
    # NxN box, for odd N:
    def avg_kernel(N):
        assert N%2 == 1
        k = np.ones((N,N), dtype=float)
        k[N/2, N/2] = 0
        k /= (N**2 - 1)
        return k

    # let's calculate one more and look and display them for comparison
    avg15 = convolve2d(imgs[0], avg_kernel(15), mode='same')

    print "Drawing L channel and local-average filtered results..."
    # you should notice that the average-filtered image maps are much
    # more smoother/blurry, as you'd expect.
    pl.figure()
    pl.imshow(avg3)
    pl.title("img0 RAW (avg3)")
    pl.colorbar()
    pl.draw()
    pl.figure()
    pl.imshow(avg15)
    pl.title("img0 RAW (avg15)")
    pl.colorbar()
    pl.draw()

    # one last tip: if you want to do root-like selections, it is possible but
    # more awkward in numpy. for example, let's only plot pixels that are
    # the location of pixels that are two standard-deviations over their
    # mean value.
    # one way is to show the image and just set all the "non-selected" pixels
    # values to zero
    print "Drawing pixels that are 2-sigma above their mean"
    pl.figure()
    pl.imshow( imgs[0] * ( (imgs[0]-pix_avg) > 2*np.sqrt(pix_var) ) )
    pl.colorbar()
    pl.title("img0 pixels over 2-sigma")
    pl.draw()

    # another way (more analogous to what ROOT does) is to make a 2d histogram
    # of pixels satisfying this criterion:

    # first, get a list of (x,y) postiions of pixels above the threshold:
    pix_2sigma = np.argwhere( (imgs[0]-pix_avg) > 2*np.sqrt(pix_var) )
    print "Found %d pixels above 2sigma threshold." % len(pix_2sigma)
    # then make a 2d histogram of these:
    pl.figure()
    pl.hist2d( pix_2sigma[:,0], pix_2sigma[:,1], bins=200)
    pl.title("img0 pixels over 2-sigma")
    pl.draw()

    print "Okay, done calculating/plotting everything!"
    print "Note that you can use the zoom tool in the GUI to interactively 'explore' the images."
    print

    raw_input("Press enter to exit...")
