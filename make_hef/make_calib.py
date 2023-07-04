import numpy as np
import glob
from PIL import Image

images = []
for f in glob.glob('imagenette2/**/*.JPEG', recursive=True)[:1024]:
    image = Image.open(f).convert('RGB').resize((640, 640))
    images.append(np.asarray(image).astype('float32'))
np.save('calib_set.npy', np.stack(images, axis=0))
