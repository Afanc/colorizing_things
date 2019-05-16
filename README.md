# colorizing_things
let's add some colors to this dull world.

Goal: Create an hand-by-hand network which will colorize black and white images.

## Approaches:

### Earlier approches:

In the beginning of this project, the main idea was to use an encoder to extract the main features of the grayscaled image and from those features to recreate the 2D image color space with a generator. Once colors are generated, there are mixed with the black and white to recreate a colorized image. And the discriminator is fed with this colorized image.

The first architecture was a adversial autoencoder. This network was suppose to learn by itself to recreate the color space of the image using the adversial process of a GAN. The color space used was the CIE Lab color space, and the generator was creating the a, b dim. The a, b are then merge with the L (grayscaled image) to create the final image.

The Encoder was a pretrained vgg16, modified to accept 1 dim in input(image in grayscale). It sends the data in a latent space Z of dim 100.

The Generator took a latent space of 100 and recreate the a, b dim of the color space. The architecture of the generator was stack of upsampling convTranspose2d to recreate the a, b dim with the correct width and height.

The Discriminator took the colorized image and tried to say if it was a real image or a fake.



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
