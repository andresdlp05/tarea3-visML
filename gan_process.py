import os
import numpy as np
import time
from PIL import Image

from tqdm import tqdm

import torch

from models import MODEL_ZOO
from models import build_generator

from umap import UMAP

# if you have CUDA-enabled GPU, set this to True!
is_cuda = False

# StyleGAN tower
model_name = 'stylegan_tower256'
model_config = MODEL_ZOO[model_name].copy()
url = model_config.pop('url')  # URL to download model if needed.

# generator
generator = build_generator(**model_config)

# load the weights of the generator
checkpoint_path = os.path.join('checkpoints', model_name+'.pth')
checkpoint = torch.load(checkpoint_path, map_location='cpu')
if 'generator_smooth' in checkpoint:
    generator.load_state_dict(checkpoint['generator_smooth'])
else:
    generator.load_state_dict(checkpoint['generator'])
if is_cuda:
    generator = generator.cuda()
generator.eval()


def sample_generator():
    code = torch.randn(1,generator.z_space_dim)
    if is_cuda:
        code = code.cuda()

    with torch.no_grad():
        # truncated normal distribution, no random noise in style layers!
        gen_out =  generator(code, trunc_psi=0.7,trunc_layers=8,randomize_noise=False)

        act2 = gen_out['act2'][0].detach()
        act3 = gen_out['act3'][0].detach()
        act3_up = torch.nn.functional.interpolate(act3.unsqueeze(0),scale_factor=2,mode='bilinear',align_corners=True)[0]
        act4 = gen_out['act4'][0].detach()

        image = gen_out['image'][0].detach()
    #

    return act2,act3,act3_up,act4,image

def postprocess(images):
    scaled_images = (images+1)/2
    np_images = 255*scaled_images.numpy()
    np_images = np.clip(np_images + 0.5, 0, 255).astype(np.uint8)
    np_images = np_images.transpose(0, 2, 3, 1)
    return np_images


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    C1, H, W = mask_a.shape
    C2, _, _ = mask_b.shape

    Ma = mask_a.reshape(C1, H * W).astype(np.uint8)  
    Mb = mask_b.reshape(C2, H * W).astype(np.uint8)  
    inter = Ma.dot(Mb.T).astype(np.float32)          

    # cardinalidades por canal
    sumA = Ma.sum(axis=1, keepdims=True).astype(np.float32)  
    sumB = Mb.sum(axis=1, keepdims=True).astype(np.float32)  
    sumB = sumB.T                                            

    union = sumA + sumB - inter                              
    # Evitar división por cero
    union = np.where(union == 0, 1.0, union)
    iou_mat = inter / union                                   

    return iou_mat

def threshold(acts: np.ndarray, k: int = 4) -> np.ndarray:
    n, C, H, W = acts.shape
    masks = np.zeros((n, C, H, W), dtype=bool)
    for c in range(C):
        all_vals = acts[:, c, :, :].reshape(-1)  
        thresh = np.quantile(all_vals, 1.0 - 1.0 / k)
        masks[:, c, :, :] = (acts[:, c, :, :] >= thresh)
    return masks 


if __name__ == '__main__':
    n_samples = 20   
    k_quantile = 4   

    STATIC_DIR   = 'static'
    IMAGES_DIR   = os.path.join(STATIC_DIR, 'images')
    IOU_DIR      = os.path.join(STATIC_DIR, 'ious')
    UMAP_OUTFILE = os.path.join(STATIC_DIR, 'umap.npy')
    X23_OUTFILE  = os.path.join(IOU_DIR, 'X23.npy')
    X34_OUTFILE  = os.path.join(IOU_DIR, 'X34.npy')

    os.makedirs(STATIC_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(IOU_DIR, exist_ok=True)

    list_act2    = []
    list_act3    = []
    list_act3_up = []
    list_act4    = []
    list_images  = []

    print(f"[1] Muestreando {n_samples} vectores latentes y extrayendo activaciones…")
    for i in tqdm(range(n_samples)):
        act2, act3, act3_up, act4, img = sample_generator()
        list_act2.append(act2.cpu().numpy())      
        list_act3.append(act3.cpu().numpy())      
        list_act3_up.append(act3_up.cpu().numpy())
        list_act4.append(act4.cpu().numpy())      

   
        np_img = postprocess(img.unsqueeze(0).cpu())  
        pil_img = Image.fromarray(np_img[0])          
        pil_img.save(os.path.join(IMAGES_DIR, f"sample_{i:04d}.png"))
   

    all_act2    = np.stack(list_act2,    axis=0)  
    all_act3    = np.stack(list_act3,    axis=0)  
    all_act3_up = np.stack(list_act3_up, axis=0)  
    all_act4    = np.stack(list_act4,    axis=0)  

    del list_act2, list_act3, list_act3_up, list_act4

 
    print("[2] Calculando max‐pooling sobre act4 y ejecutando UMAP…")
    features = all_act4.max(axis=2).max(axis=2)  
    reducer = UMAP(n_components=2, random_state=42)
    projections = reducer.fit_transform(features)  
    np.save(UMAP_OUTFILE, projections)
    print(f"  → Proyecciones UMAP guardadas en '{UMAP_OUTFILE}'")


    print("[3] Umbralizando (quantile threshold) cada canal por capa…")
    mask2 = threshold(all_act2,    k_quantile)  
    mask3 = threshold(all_act3_up, k_quantile) 
    mask4 = threshold(all_act4,    k_quantile)  

   
    print("[4] Upsampling de mask2 (8×8 → 16×16) …")
    n, C, H2, W2 = mask2.shape          
    _, _, H3, W3 = mask3.shape          
    mask2_up = np.zeros((n, C, H3, W3), dtype=bool)

    for i in range(n):
        for c in range(C):
       
            arr = (mask2[i, c].astype(np.uint8) * 255)
            pil = Image.fromarray(arr, mode='L').resize((W3, H3), resample=Image.NEAREST)
            mask2_up[i, c] = (np.array(pil) >= 128)

    print("[5] Calculando IoU 2→3 por muestra …")
    X23 = np.zeros((n, C, C), dtype=np.float32) 
    for i in tqdm(range(n)):
        X23[i] = iou(mask2_up[i], mask3[i])

    print("[6] Calculando IoU 3→4 por muestra …")
    X34 = np.zeros((n, C, C), dtype=np.float32) 
    for i in tqdm(range(n)):
        X34[i] = iou(mask3[i], mask4[i])

    np.save(X23_OUTFILE, X23)
    np.save(X34_OUTFILE, X34)
    print(f"  → X23 guardado en '{X23_OUTFILE}'")
    print(f"  → X34 guardado en '{X34_OUTFILE}'")

    print("Preprocesamiento completado. Archivos escritos en 'static/'.")
