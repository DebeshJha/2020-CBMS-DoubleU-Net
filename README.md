# DoubleU-Net
DoubleU-Net starts with a VGG19 as encoder sub-network, which is followed by decoder sub-network. In the network, the input image is fed to the modified UNet(UNet1), which generates predicted masks (i.e., output1). We then multiply the input image and the produced masks (i.e., output1), which acts as an input for the second modified U-Net(UNet2) that produces another the generated mask (output2). Finally, we concatenate both the masks (output1 and output2) to get the final predicted mask (output). <br/>

## Architecture
<img src="img/DoubleU-Net.png">
