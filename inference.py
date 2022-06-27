import os
import numpy as np
import tensorflow as tf
from tensorflow.math import add as Add
from tensorflow.math import multiply as Multiply
from tensorflow.math import exp as Exp
from model import Generator
from utils import load_image, imsave, imresize, normalize_m11, create_dir


class Inferencer:
    def __init__(self,
                 num_samples,
                 scale_factor,
                 inject_scale,
                 result_dir,
                 checkpoint_dir):

        self.model = []
        self.NoiseAmp = []
        self.load_model(checkpoint_dir)
        self.num_samples = num_samples
        self.scale_factor = scale_factor
        self.inject_scale = inject_scale
        self.result_dir = result_dir


    def load_model(self, checkpoint_dir):
        """ Load generators and NoiseAmp from checkpoint_dir """
        self.NoiseAmp = np.load(checkpoint_dir + '/NoiseAmp.npy')
        _dir = os.walk(checkpoint_dir)
        for path, dir_list, _ in _dir:
            for dir_name in dir_list:
                network = dir_name[0]
                scale = int(dir_name[1])
                if network == 'G':
                    generator = Generator(num_filters=32*pow(2, (scale//4)))
                    generator.load_weights(os.path.join(path, dir_name) + '/G').expect_partial()    # Silence the warning
                    self.model.append(generator)


    def inference(self, mode, reference_image, image_size=250):
        """ Use SinGAN to do inference
        mode : Inference mode
        reference_image : Input image name
        image_size : Size of output image
        """
        reference_image = load_image(reference_image, image_size=image_size)
        reference_image = normalize_m11(reference_image)
        reference_image = tf.math.multiply(tf.cast(tf.math.greater(reference_image, -1), tf.float32), reference_image)
        reals = self.create_real_pyramid(reference_image, num_scales=len(self.model))

        _dir = create_dir(os.path.join(self.result_dir, mode))
        if mode == 'random_sample':
            z_fixed = tf.random.normal(reals[0].shape)
            for n in range(self.num_samples):
                fake = self.SinGAN_generate(reals, z_fixed, inject_scale=self.inject_scale)
                imsave(fake, _dir + f'/random_sample_{n}.jpg') 

        elif (mode == 'harmonization') or (mode == 'editing') or (mode == 'paint2image'):
            fake = self.SinGAN_inject(reals, inject_scale=self.inject_scale)
            imsave(fake, _dir + f'/inject_at_{self.inject_scale}.jpg')
        
        elif mode == 'sparse_inject':
            fake = self.SinGAN_sparse_inject(reals, inject_scale=self.inject_scale)
            imsave(fake, _dir + f'/spare_inject_at_{self.inject_scale}.jpg')

        else:
            print('Inference mode must be: random_sample, harmonization, paint2image, editing')


    def SinGAN_inject(self, reals, inject_scale=1):
        """ Inject reference image on given scale (inject_scale should > 0)"""
        fake = reals[inject_scale]

        for scale in range(inject_scale, len(reals)):
            fake = imresize(fake, new_shapes=reals[scale].shape)
            z = tf.random.normal(fake.shape)
            z = z * self.NoiseAmp[scale]
            fake = self.model[scale](fake, z)
    
        return fake
    
#     @tf.function
    def SinGAN_sparse_inject(self, reals, inject_scale):
        """ Inject reference data at given scale after normal random generation"""
        fake = tf.zeros_like(reals[0])
        num_scales = len(reals)
        
        for scale, generator in enumerate(self.model):
            if scale >= inject_scale:
                amplitude = self.inject_amp(scale, num_scales)
#                 Get inverse binary mask of values in sparse data
                mask = tf.cast(tf.math.logical_not(tf.math.greater(reals[scale], 0)), tf.float32)
#                 Adjust amplitude of mask for future injection
                mask = Add(mask, Multiply(tf.cast(1 - amplitude, tf.float32), tf.cast(tf.math.greater(reals[scale], 0), tf.float32)))
                fake = imresize(fake, new_shapes=reals[scale].shape)
#                 Inject with amplification adjustment
                fake = Add(Multiply(mask, fake), Multiply(tf.cast(amplitude, tf.float32), reals[scale]))
            else:    
                fake = imresize(fake, new_shapes=reals[scale].shape)
            
            z = tf.random.normal(fake.shape)
            z = z * self.NoiseAmp[scale]
            fake = generator(fake, z)
            self.save_scale(fake, scale)
    
        return fake
    
    def save_scale(self, fake, scale):
        _dir = os.path.join(self.result_dir, 'sparse_inject')
        imsave(fake, _dir + f'/scaled_injection_scale_{scale}.jpg')
    
    def inject_amp(self, scale, num_scales):
#         Essentially flipped sigmoid curve centered at middle of num_scales
        return -1*(1 / (1 + Exp(-(1.0*scale - .5*num_scales)))) + 1

    @tf.function
    def SinGAN_generate(self, reals, z_fixed, inject_scale=0):
        """ Use fixed noise to generate before start_scale """
        fake = tf.zeros_like(reals[0])
    
        for scale, generator in enumerate(self.model):
            fake = imresize(fake, new_shapes=reals[scale].shape)
            
            if scale > 0:
                z_fixed = tf.zeros_like(fake)

            if scale < inject_scale:
                z = z_fixed
            else:
                z = tf.random.normal(fake.shape)

            z = z * self.NoiseAmp[scale]
            fake = generator(fake, z)

        return fake


    def create_real_pyramid(self, real_image, num_scales):
        """ Create the pyramid of scales """
        reals = [real_image]
        for i in range(1, num_scales):
            reals.append(imresize(real_image, scale_factor=pow(self.scale_factor, i)))
        
        """ Reverse it to coarse-fine scales """
        reals.reverse()
        for real in reals:
            print(real.shape)
        return reals