# colorizing_things
let's add some colors to this dull world

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
