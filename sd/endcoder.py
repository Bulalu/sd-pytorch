import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    """
        This code is for a machine (called VAE_Encoder) that takes a picture and changes it step-by-step into a simpler, smaller version.
        It keeps all the important parts of the picture while making it easy for a computer to use.
        This smaller version is then ready for the computer to do some magic and make new pictures that look like the original.
    """
    def __init__(self):
        super().__init__(
            # First layer: Takes a colorful image and starts to simplify it, keeping important features.
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            
            # These blocks help remember and enhance important details like edges and textures in the image.
            VAE_ResidualBlock(128, 128),  # Keeps working on the simplified image.
            VAE_ResidualBlock(128, 128),  # Keeps working on the simplified image.
            
            # This layer makes the image smaller, like zooming out, but still keeps the important parts.
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            
            # More blocks that keep enhancing the important details in the smaller image.
            VAE_ResidualBlock(128, 256),  # Enhances details in a smaller image.
            VAE_ResidualBlock(256, 256),  # Enhances details in a smaller image.
            
            # Another layer that further zooms out the image, making it even smaller.
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0), 
            
            # These blocks continue to focus on the important parts of the now smaller image.
            VAE_ResidualBlock(256, 512),  # Focuses on important parts.
            VAE_ResidualBlock(512, 512),  # Focuses on important parts.
            
            # Zooms out the image again, making it smaller and more simplified.
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0), 
            
            # Final blocks that ensure the most important details of the image are still captured.
            VAE_ResidualBlock(512, 512),  # Captures essential details.
            VAE_ResidualBlock(512, 512),  # Captures essential details.
            VAE_ResidualBlock(512, 512),  # Captures essential details.
            
            # This special block helps the model pay attention to the very important parts of the image.
            VAE_AttentionBlock(512), 
            
            # One more block to refine the details of the image.
            VAE_ResidualBlock(512, 512), 
            
            # Normalizes the image data, making sure it's not too bright or too dark.
            nn.GroupNorm(32, 512), 
            
            # Adds a bit of complexity back to the simplified image to make it more realistic.
            nn.SiLU(), 

            # These layers further simplify the image into a very small but information-rich version.
            nn.Conv2d(512, 8, kernel_size=3, padding=1),  # Makes the image much simpler.
            nn.Conv2d(8, 8, kernel_size=1, padding=0),   # Final touch on the simplified image.
        )

    def forward(self, x, noise):
        # x is the image we put in, and noise is like a little bit of magic dust.

        for module in self:
            # If we're zooming out the image, we add a tiny bit of space around it.
            if getattr(module, 'stride', None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))  # Add space to the right and bottom sides.

            x = module(x)  # Pass the image through each magical transformation layer.

        # The image is now split into two magical parts: one part tells us where the image is bright or dark,
        # and the other part tells us how much variety there is in different parts of the image.
        mean, log_variance = torch.chunk(x, 2, dim=1)

        # We make sure the variety part is not too crazy or too boring.
        log_variance = torch.clamp(log_variance, -30, 20)

        # Convert the variety part into a form that we can work with.
        variance = log_variance.exp()
        stdev = variance.sqrt()
        
        # We mix the bright-dark part with a bit of the magical dust using the variety part.
        x = mean + stdev * noise
        
        # Finally, we adjust the image just a bit to make it perfect.
        x *= 0.18215
        
        return x
