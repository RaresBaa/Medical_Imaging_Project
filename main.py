import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def cmb(path, angle):
    """Combines all of the preprocessing functions"""
    #we import the image as a numpy array
    image = np.array(Image.open(path))
    #we convert the image to grayscale
    grayscale_img = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    #we rotate the image to fit better in the frame
    rotated_img = np.array(Image.fromarray(grayscale_img).rotate(angle))
    #we remove the black background that remains after the rotation
    removed_black_img = np.where(rotated_img <= 20, 255, rotated_img)
    return removed_black_img

def show_all_images(title, eosinophil, lymphocyte, monocyte, neutrophil):
    """Shows all the images in a single figure"""
    plt.figure()
    plt.suptitle(title)
    for i in range(3):
        plt.subplot(3, 4, i * 4 + 1)
        plt.imshow(eosinophil[i], cmap='gray')
        plt.axis('off')
        plt.title('Eosinophil' + str(i + 1))
        plt.subplot(3, 4, i * 4 + 2)
        plt.imshow(lymphocyte[i], cmap='gray')
        plt.axis('off')
        plt.title('Lymphocyte' + str(i + 1))
        plt.subplot(3, 4, i * 4 + 3)
        plt.imshow(monocyte[i], cmap='gray')
        plt.axis('off')
        plt.title('Monocyte' + str(i + 1))
        plt.subplot(3, 4, i * 4 + 4)
        plt.imshow(neutrophil[i], cmap='gray')
        plt.axis('off')
        plt.title('Neutrophil' + str(i + 1))
    plt.show()

def show_image_set(title, arr, original):
    """Shows a single type of cell"""
    plt.figure()
    plt.suptitle(title)
    for i in range(3):
        plt.subplot(2, 3, i + 1)
        plt.imshow(arr[i], cmap='gray')
        plt.axis('off')
        plt.title(str(i + 1))
        plt.subplot(2, 3, i + 4)
        plt.imshow(original[i], cmap='gray')
        plt.axis('off')
        plt.title('Original' + str(i + 1))
    plt.show()

def contrast_enhancement(original_image):
    """Returns the contrast enhanced image"""
    in_a = 150
    in_b = 150
    out_a = 0
    out_b = 255

    processed_image = original_image.astype(float)
    s = original_image.shape
    for i in range(s[0]):
        for j in range(s[1]):
            if original_image[i][j] < in_a:
                processed_image[i, j] = (out_a * original_image[i][j]) / in_a
            elif in_a <= original_image[i][j] < in_b:
                processed_image[i, j] = out_a + ((out_b - out_a) * (original_image[i, j] - in_a)) / (in_b - in_a)
            elif original_image[i][j] >= in_b:
                processed_image[i, j] = out_b + ((255 - out_b) * (original_image[i, j] - in_b)) / (255 - in_b)

    processed_image = np.clip(processed_image, 0, 255)
    processed_image = processed_image.astype('uint8')

    return 255 - processed_image

def binary_thresholding(img):
    """Apply binary thresholding to an image."""
    threshold = 150
    processed_image = (img >= threshold) * 255
    return 255 - processed_image.astype('uint8')

def power_function(original_image):
    power_coef = 2.5
    processed_image = original_image.astype(float)
    processed_image = 255 * (original_image / 255) ** power_coef
    processed_image = np.clip(processed_image, 0, 255)
    processed_image = processed_image.astype('uint8')

    return 255 - processed_image

def main():
    #import all of the images, with the preprocessing done
    eosinophil = [cmb('Eosinophil_1.jpeg', -2), cmb('Eosinophil_2.jpeg', -15), cmb('Eosinophil_3.jpeg', -5)]
    lymphocyte = [cmb('Lymphocyte_1.jpeg', 15), cmb('Lymphocyte_2.jpeg', 5), cmb('Lymphocyte_3.jpeg', 5)]
    monocyte = [cmb('Monocyte_1.jpeg', 5), cmb('Monocyte_2.jpeg', -5), cmb('Monocyte_3.jpeg', 2)]
    neutrophil = [cmb('Neutrophil_1.jpeg', -5), cmb('Neutrophil_2.jpeg', 13), cmb('Neutrophil_3.jpeg', 0)]

    #first we show the original data
    show_all_images( "Original Data", eosinophil, lymphocyte, monocyte, neutrophil)

    #for each type of cell we apply each transformation and show the results
    eosinophil_processed = [contrast_enhancement(eosinophil[0]), binary_thresholding(eosinophil[1]), power_function(eosinophil[2])]
    lymphocyte_processed = [contrast_enhancement(lymphocyte[0]), binary_thresholding(lymphocyte[1]), power_function(lymphocyte[2])]
    monocyte_processed = [contrast_enhancement(monocyte[0]), binary_thresholding(monocyte[1]), power_function(monocyte[2])]
    neutrophil_processed = [contrast_enhancement(neutrophil[0]), binary_thresholding(neutrophil[1]), power_function(neutrophil[2])]
    #we show the results
    show_all_images( "Processed", eosinophil_processed, lymphocyte_processed, monocyte_processed, neutrophil_processed)


    # apply the same transformations to the same cell type
    #for Eosinophil cells we apply the contrast enhancement method
    eosinophil_contrasted = [contrast_enhancement(eosinophil[0]), contrast_enhancement(eosinophil[1]), contrast_enhancement(eosinophil[2])]
    #for Lymphocyte cells we apply the binary thresholding method
    lymphocyte_threshold = [binary_thresholding(lymphocyte[0]), binary_thresholding(lymphocyte[1]), binary_thresholding(lymphocyte[2])]
    #for Monocyte cells we apply the power function method
    monocyte_power = [power_function(monocyte[0]), power_function(monocyte[1]), power_function(monocyte[2])]

    #we show the results
    show_image_set("Eosinophil - Contrast Enhancement", eosinophil_contrasted, eosinophil)
    show_image_set("Lymphocyte - Binary Thresholding", lymphocyte_threshold, lymphocyte)
    show_image_set("Monocyte - Power Function", monocyte_power, monocyte)

if __name__ == "__main__":
    main()