#!/usr/bin/env python

from PIL import Image
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

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Example for processing jpeg images")

    # protip: nargs="+" means accept one or more arguments from the command
    # line; they will be given to us as a list regardless.
    parser.add_argument("input_image", nargs="+", help="input jpeg file(s)")

    args = parser.parse_args()

    # open up all the input images
    print "Opening %d files..." % len(args.input_image)
    imgs_original = [Image.open(f) for f in args.input_image]

    # convert each image into a numpy array; the PIL library
    # has made it so that all you have to do is "cast" the Image
    # object to a numpy array and it will do the right thing.
    print "Converting images to numpy arrays..."
    imgs = [np.array(img) for img in imgs_original]

    # Ok, right now each image array has the shape (w,h,3);
    # so an R,G,B tuple at each pixel. and we have a _list_
    # of these arrays.
    
    # Let's just turn the whole thing into
    # a giant 4-d array. This will make some calculations
    # easiers to express and/or faster, but note that if you
    # try to load too many images it might get to be too much memory
    # to allocate into one giant array. That's because numpy
    # arrays are actually an interface to native C arrays, and those
    # are contiguous in memory. Wheras python _lists_ are more like
    # a linked list and do not require contiguous memory (which is
    # why many of their operations are slower to begin with...)

    imgs = np.array(imgs)

    # okay so now we have an array with shape (N, w, h, 3), where
    # N is the number of images passed in on the command line.
    

    # Lets filter it out  into separate arrays for each channel:

    print "Splitting up R,G,B..."
    imgs_r = imgs[:,:,:,0]
    imgs_g = imgs[:,:,:,1]
    imgs_b = imgs[:,:,:,2]

    # now we have (N, w, h) shaped arrays; one value for each pixel
    # of each image, split out into 3 different channels.
    # let's just display a preview of the first image in each channel:
    print "Drawing image channels"
    pl.figure()
    pl.imshow(imgs_r[0])
    pl.colorbar()
    pl.title("Red")
    pl.draw() # force immmediate graphics update when running in commandline mode

    pl.figure()
    pl.imshow(imgs_g[0])
    pl.colorbar()
    pl.title("Green")
    pl.draw()

    pl.figure()
    pl.imshow(imgs_b[0])
    pl.colorbar()
    pl.title("Blue")
    pl.draw()

    # at this point we could calculate our own luminance value, if
    # we wanted. For example a (naive) guess of how to calculate
    # the white level could be to just average R, G and B:

    imgs_L_naive = 1/3. * (imgs_r + imgs_g + imgs_b)

    # In reality, the luminiance calulcation is a bit more nuanced.
    # PIL already knows how to do it so let's just ask it to convert
    # our images to L channel. To do this we go back to original PIL
    # image objects (not the numpy arrays we converted them into) and
    # use the convert() function, casting them into numpy arrays on the
    # fly:
    imgs_L = np.array([np.array(img.convert("L")) for img in imgs_original])

    # let's see what the RMS difference in our naive vs. real L channel
    # looks like, for the first image:
    print "RMS difference for L:", np.sqrt(np.mean((imgs_L_naive[0] - imgs_L[0])**2))

    # and let's just histogram the different channels for the first image
    # note that we want to flatten the image, i.e. crush the 2d info and
    # turn our (w,h)-shaped array into a 1d array of lenght w*h. If we don't,
    # the histogram function will try to make `w` histograms of `h` elements each.
    pl.figure()
    xmax = int(np.ceil(imgs[0].max()))
    pl.hist(imgs_r[0].flat, histtype='step', bins=xmax, range=(0,xmax), color='r', label='r')
    pl.hist(imgs_g[0].flat, histtype='step', bins=xmax, range=(0,xmax), color='g', label='g')
    pl.hist(imgs_b[0].flat, histtype='step', bins=xmax, range=(0,xmax), color='b', label='b')
    pl.hist(imgs_L[0].flat, histtype='step', bins=xmax, range=(0,xmax), color='black', label='L (PIL)')
    pl.hist(imgs_L_naive[0].flat, histtype='step', bins=xmax, range=(0,xmax), color='magenta', label='L (naive)')
    pl.title('image 0 channel statistics')
    pl.xlabel('pix val')
    pl.legend(loc='best')
    pl.draw()

    # About calculating statistics on numpy arrays: by default numpy will
    # run over the entire array in all dimensions. So to get the average
    # value of all pixels in all images, you could do:
    print "Avg L in all pixels and images:", imgs_L.mean()

    # but that's not very meaningful. let's instead calculate the average
    # value of each image in sequence; to do so, tell numpy to use only the
    # trailing two axes:
    print "average L in each image: ", imgs_L.mean(axis=(1,2))
    # note that this gives you an array with shape (N,); one number
    # for each of the N images.

    # another really cool thing we can do is average each individual pixel
    # over the sequence of images by telling numpy to average over axis=0:
    pix_avg = imgs_L.mean(axis=0)
    # now we should have an array with the shape (w,h), i.e. the original
    # dimensions of the jpg images, and each entry in the array is that
    # particular pixel's average over the N input images. we can do the
    # same thing with variance:
    pix_var = imgs_L.var(axis=0)
    # and maximums:
    pix_max = imgs_L.max(axis=0)

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
    avg3 = convolve2d(imgs_L[0], k3, mode='same')

    # while we're at it, we could generically caluclate the average for any
    # NxN box, for odd N:
    def avg_kernel(N):
        assert N%2 == 1
        k = np.ones((N,N), dtype=float)
        k[N/2, N/2] = 0
        k /= (N**2 - 1)
        return k

    # let's calculate one more and look and display them for comparison
    avg15 = convolve2d(imgs_L[0], avg_kernel(15), mode='same')

    print "Drawing L channel and local-average filtered results..."
    # you should notice that the average-filtered image maps are much
    # more smoother/blurry, as you'd expect.
    pl.figure()
    pl.imshow(imgs_L[0])
    pl.title("img0 L-channel")
    pl.colorbar()
    pl.draw()
    pl.figure()
    pl.imshow(avg3)
    pl.title("img0 L-channel (avg3)")
    pl.colorbar()
    pl.draw()
    pl.figure()
    pl.imshow(avg15)
    pl.title("img0 L-channel (avg15)")
    pl.colorbar()
    pl.draw()

    # one last tip: if you want to do root-like selections, it is possible but
    # more awkward in numpy. for example, let's only plot pixels that are
    # the location of pixels that are two standard-deviations over their
    # mean value.
    # one way is to show the image and just set all the "non-selected" pixels
    # values to zero, by multiplying the boolean result (0 or 1) of the selection:
    print "Drawing pixels that are 2-sigma above their mean"
    pl.figure()
    pl.imshow( imgs_L[0] * ( (imgs_L[0]-pix_avg) > 2*np.sqrt(pix_var) ) )
    pl.colorbar()
    pl.title("img0 pixels over 2-sigma")
    pl.draw()

    # another way (more analogous to what ROOT does) is to make a 2d histogram
    # of pixels satisfying this criterion:

    # first, get a list of (x,y) postiions of pixels above the threshold:
    pix_2sigma = np.argwhere( (imgs_L[0]-pix_avg) > 2*np.sqrt(pix_var) )
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
