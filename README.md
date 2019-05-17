# colorizing_things
let's add some colors to this dull world.

Groupe members: ...,Anthony Gillioz

## Objectives:
Goal: Create an hand-by-hand network which will colorize black and white images.

## Problematic:

## Approaches:

### First approches:

At the beginning of this project, the main idea was to use an encoder to extract the main features of a grayscaled image, and from those features to recreate a 2D colored image with a generator/decoder. Once the colors generated, those would be mixed with the grayscale image to recreate a colored image.
Since we didn't want this autoencoder to generate images "as close as possible" to the actual image, since many different color combinations can be plausible for a single grayscaled image, we went on using a GAN architecture, with the discriminator fed with fake colorized and real color images.

So the first architecture was a adversial autoencoder. This network was supposed to learn by itself to recreate the color space of the image using the adversial process of a GAN. The color space used was the CIE Lab color space, and the generator was creating the a, b dim. The a, b are then merge with the L (grayscaled image) to create the final image.

The Encoder was a pretrained vgg16, modified to accept 1 dim in input (image in grayscale). It sends the data in a latent space Z of dim 100.
The Generator took a latent space of 100 and recreated the a, b dim of the color space. The architecture of the generator was stacks of upsampling convTranspose2d layers to recreate the a, b dim with the correct width and height.
The Discriminator took the colorized image and tried to say if it was a real image or a fake was build in symmetry of the generator, as stacks of conv2d layers.

The first networks were first tested with a vanilla GAN loss (BCELoss), then with LSGAN and Wasserstein Loss (loss functions used in this project are explained in the Sec. Loss tested).
When using BCELoss an LSGAN, we could not make the network converge. This, of course, resulted in noisy images.

![Very first results](imgs/very_first_results.jpg)

Using Wasserstein Loss, we were able to produce actual patterns in images but no matter the parameters and the length of the training, results were always of poor quality (see images below), with little to no consistency over time.

![First results](imgs/first_results.jpg)

We moved on to adding self-attention layers, as described by Zhang et al. (2018) [https://arxiv.org/abs/1805.08318] in both the generator and discriminator, as penultimate layers, implementing Hinge Loss on the go as described by Zhang et al. This resulted in our first results where the generatoractually took account of the images' edges after 40'000 iterations.

![SAGAN 1](imgs/res_sagan1.jpg)

However, no big improvement was seen during the next 350'000 iterations whereas implausible patches continued appearing on certain images and some features, like grass, seemed to be very hard to colorize.

![SAGAN 2](imgs/res_sagan2.jpg)

At this point we tested multiple strategies to improve our network.
The first thing was to deepen the network, going to 19 layers on the generator with the pretrained weights from vgg19.
### difference ?
We tried :
    - training the discriminator more than the generator, at different rates
    - adding residual connections to the generator
    - outputing colors in the cielab colorspace and feeding this to the discriminator
    - implementing a VAE instead of AE, introducing noise before decoding
    - training the autoencoder to generate grayscales before training it to colorize images
    - implementing a shading autoencoder as described by K. Frans (2017) [https://arxiv.org/abs/1704.08834] parallel to our generator, feeding both networks to the discriminator

Results for all attempts are not shown as they generally led to worse results than with our final architecture. When results were similar and a choice had to be made between two networks, the least complex and the most memory efficient approach was always selected. In general, all networks seemed to learn to colorize images after iterating through ~1 milion images (variable batch sizes), yet most networks did (one of the following) :
    - not perform with consistency (oscillation)
    - collapsed (when using "traditional losses" such as MSE, BCE)
    - failed to sharpen colorization around the edges
    - failed to remove implausible patches of colors
    - failed to perform correctly on all types of images 

######### when did we start producing rgb's again ?

### Final approach :




#### Loss tested:

1. LSGAN:
2. Wasserstein GAN:
3. Adversial hinge loss:

structure :
VAE into a GAN
https://imgur.com/a/hpZLm99
-> 
    base_image -> grayscale -> bigboy152/smallboy16 but smallboy16 first (VGG16) (modified because takes rgb) -> encoder -> discriminator
    -> so discriminator receives from encoder 1/2 or base_image 1/2
    -> decoder does it in 2 dimensions (cielab), we sum it to grayscale to get proposed colorized
    -> we could, later on, augment the dataset based on the latent space of the vae

TODO :
    -> implement the vae
        -> implement the smallboy16
            -> grayscale first
            -> modify smallboy to understand b_and_w
        -> decoder
            -> deconv. into 2 channels
        -> join it with the b_and_w channel

    -> gan
        -> in theory, generator and discriminator are symmetric 
        -> and shit anthony how to do 


SOTA:
1. https://richzhang.github.io/colorization/resources/colorful_eccv2016.pdf
2. http://cs231n.stanford.edu/reports/2016/pdfs/219_Report.pdf
3. http://openaccess.thecvf.com/content_cvpr_2017/papers/Deshpande_Learning_Diverse_Image_CVPR_2017_paper.pdf
