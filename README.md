**Summary:** This repository contains relevant code created for the paper "Label-efficient phylloxera gall segmentation via 3D bottom-to-top leaf registration" by Eric Stumpe, Zeinab Maleki Asayesh, Astrid Forneck and Matthias Zeppelzauer. The accompanying data can be found under: https://zenodo.org/records/20795851

**Paper Abstract:** Grape phylloxera (Daktulosphaira vitifoliae Fitch) is a destructive pest that forms galls on grapevine leaves and roots, posing a major threat to vineyard productivity. While deep learning methods have shown promise for plant disease detection, accurate segmentation of individual phylloxera galls remains challenging. Galls are best visible on the bottom (abaxial) leaf side. For plant monitoring, however, the detection typically has to be performed from the top  (adaxial)  side, which is challenging because the visual symptoms are less pronounced.
We address the problem of label-efficient gall segmentation across leaf sides by learning gall detection on the bottom side and transferring this knowledge for detection and segmentation from the top side. To this end, we propose a 3D raycasting-based image registration method for aligning top and bottom leaf images, enabling label transfer across leaf sides. To our knowledge, this is the first dedicated approach for transfer-learning a vision model from the bottom to the top side of a leaf. For comparison, we investigate unsupervised domain adaptation (UDA) techniques that treat bottom images as a labeled source domain and top images as an unlabeled target domain. We further introduce label refinement strategies to mitigate registration errors.
Experiments on hyperspectral grapevine leaf data show that models trained with transferred and refined labels are on par and partly even better than a fully supervised baseline trained on hand-labeled images from the top side (peak F1 of 0.771 for transfer-learned model vs. 0.739 for fully supervised baseline).
Our results demonstrate under controlled lab conditions that phylloxera gall detection and segmentation can be achieved from the top side of the leaf without requiring dedicated top-side annotations. Easy-to-achieve bottom-side annotations are sufficient for transfer learning robust models. Our work has practical implications, as reliable detection and segmentation of galls from the top side facilitates label-efficient plant monitoring.

**Code Overview:**

- **transfer_masks.py:** with this code, in a loop, hand annotated gall masks are transferred from the bottom leaf side to the top leaf side via our developed registration algorithm. 
In addition, hyperspectral images can be registered from the top to the bottom side.
- **refined_mask_generation.py:** the transferred gall masks are refined with the option of dilation or peak detection
- **new_run_loop.py:** here, all experimental settings for training the models presented in the paper can be set up
- **config_presets.py:** here, standard experimental settings are defined to keep new_run_loop.py leaner
- **main_code.py:** is started from new_run_loop.py and executes the DL learning schedule
- **utils.py** contains all helper functions needed for executing the DL-training methods

The data repository (https://zenodo.org/records/20795851) contains the following directories needed to run the code:

**dataset** - contains all recorded multimodal camera data of grapevine leaves infested by phylloxera, which were captured from the top and bottom sides. In addition, masks used for training DL-based segmentation are also part of this directory.

- **hs_np:** contains all recorded hyperspectral images (SpecimIQ) in .npy format
- **hs:** contains the same hyperspectral images as RGB representation in .png format
- **rgb:** contains RGB images recorded with an Orbbec Femto Bolt camera in .png format
- **infrared:** contains infrared images from the Orbbec Femto Bolt camera in .npy format
- **depth:** contains depth maps from the Orbbec Femto Bolt camera in .npy format
- **thermal:** contains thermal images recorded with a VarioCAM HDx Head 620 S thermal camera in .npy format (not used in publication)
- **hs_mask_annot_bot:** contains a binary image mask with hand-labeled phylloxera galls of bottom side leaf images
- **hs_mask_annot_top:** contains a binary image mask with hand-labeled phylloxera galls of top side leaf images
- **hs_mask_transfer_direct:** contains gall segment mask that have been transferred from bottom to top side with the developed registration algorithm
- **hs_mask_transfer_dilated:** contains gall segmentation masks that have been refined via dilation from hs_mask_transfer_direct
- **hs_mask_transfer_peak:** contains gall segment mask that have been refined via thresholding and pseudo labels from hs_mask_transfer_direct
- **hs_pseudo_top:** contains the raw top-side pseudo labels predicted by a Patch-CNN model that has been trained on bottom-side data (.npy)
- **hs_pseudo_top_binary:** contains hs_pseudo_top, but binarized at a value of 0.5

**calibration_params** - contains intrinsic and extrinsic camera parameters for each camera (hs: Specim IQ, rgbd: Orbbec Femto Bolt, thermal: VarioCAM HDx Head 620 S). Was calculated using checker board calibration.

**pretrained_weights** - contains model weights (.pth - PyTorch) of three different models (MLP, Patch-CNN, UNET) that have been trained on bottom-side data.
These models are then used for transfer learning.
