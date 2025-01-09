# Data extracted from the table for calculations

# Metrics: SSIM, PSNR, LPIPS
textbf_values = {
    "Cifar10": [0.9966, 43.58, 1.88e-4],
    "Gtsrb": [0.9962, 42.58, 1.54e-4],
    "Imagenette": [0.9948, 44.90, 2.55e-4],
    "CelebA": [0.9987, 51.16, 5.50e-5]
}

underline_values = {
    "Cifar10": [0.9955, 40.37, 3.35e-4],
    "Gtsrb": [0.9877, 40.84, 3.44e-4],
    "Imagenette": [0.9931, 36.23, 1.66e-3],
    "CelebA": [0.9914, 36.99, 9.75e-4]
}

# Calculate percentage improvement for each metric and dataset
improvements = {
    dataset: [
        100 * (textbf - underline) / underline 
        for textbf, underline in zip(textbf_values[dataset], underline_values[dataset])
    ]
    for dataset in textbf_values
}

print(improvements)
