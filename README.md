Sarcasm Detection Using Multimodal Analysis

## Project Overview

This project aims to detect sarcasm using a combination of **textual and visual data**. It employs state-of-the-art machine learning and deep learning techniques to analyze sarcasm within multimodal contexts. The system integrates **text processing models** (like Roberta) and **image analysis models** (like ViT - Vision Transformer) to make predictions. The pipeline also includes OCR (Optical Character Recognition) for text extraction from images, enabling deeper multimodal understanding.

---

## Key Features

1. **Multimodal Data Handling**  
   - Text features are processed using the `RobertaModel` from Hugging Face Transformers.  
   - Image features are processed using the `Vision Transformer (ViT)` from the `timm` library.

2. **OCR Integration**  
   - Utilizes Microsoft's **TrOCR model** to extract handwritten text from images.  
   - Handles cases where text length is insufficient or OCR fails.

3. **Sarcasm Detection**  
   - Textual sarcasm detection uses **Twitter Roberta Base (cardiffnlp)**, fine-tuned for the sarcasm detection task.  
   - Combines textual and visual information for robust sarcasm detection in paired inputs.

4. **Data Preprocessing**  
   - Applies image preprocessing using PyTorch’s `torchvision.transforms`.  
   - Supports transformation for train, validation, and test datasets.

5. **Dynamic and Flexible Model Architecture**  
   - The `DynRT` model combines **Roberta embeddings**, **ViT features**, and a **custom TRAR module** for multimodal reasoning.  
   - Freeze layers in base models as per the configuration to support both fine-tuning and feature extraction workflows.

6. **Error Handling**  
   - Ensures fallback mechanisms when OCR fails or no text is detected in images.  
   - Prepares fallback tensors for such cases to avoid disrupting the training process.

---

## Prerequisites

### Libraries and Dependencies
Install the following Python libraries using `pip` or `conda`:

- **PyTorch**: `torch`, `torchvision`, `timm`  
- **Hugging Face Transformers**: `transformers`  
- **OCR Tools**: `pytesseract`, `Pillow`  
- **Utilities**: `numpy`, `pickle`, `scipy`

```bash
pip install torch torchvision timm transformers pytesseract Pillow numpy scipy
```

---

## Dataset Structure

1. **Text Data**  
   - Preprocessed and stored in pickle files.  
   - Paths: `train_text`, `valid_text`, `test_text`.

2. **Image Data**  
   - Images stored in the `dataset_image/` folder.  
   - Each image is named using a unique identifier (e.g., `123.jpg`).

3. **Labels**  
   - Stored in pickle files with paths `train_labels`, `valid_labels`, and `test_labels`.

4. **Image Tensors**  
   - Processed tensors saved in the `image_tensor/` directory.

---

## Project Workflow

1. **Data Preprocessing**
   - Texts and images are loaded using pickle files.
   - Images are transformed and saved as tensors for efficient processing.

2. **OCR Processing**
   - OCR is applied to extract text from images.
   - Handles scenarios with missing or insufficient text by substituting image embeddings.

3. **Model Construction**
   - The `DynRT` model is built using the `RobertaModel` and `ViT`.
   - Layers can be frozen for efficient feature extraction.

4. **Training and Evaluation**
   - Uses the sarcasm detection model to classify multimodal input pairs.
   - Evaluates performance based on textual and visual sarcasm indicators.

---

## Directory Structure

```
project/
├── dataset_image/       # Contains image data (e.g., 123.jpg)
├── input/
│   └── prepared/        # Preprocessed pickle files (train_id, test_id, valid_id, etc.)
├── image_tensor/        # Contains processed image tensors (e.g., 123.npy)
├── scripts/             # Contains Python scripts for model building and processing
├── main.py              # Main entry point for training and evaluation
└── README.md            # Project documentation
```

---

## Usage Instructions

### 1. Prepare Data
Ensure all required datasets and labels are in the appropriate folders (`dataset_image/` and `input/prepared/`).

### 2. Run the Script
Use `main.py` to process data and train the model.

```bash
python main.py
```

### 3. Analyze Results
- Outputs will be saved in the `image_tensor/` directory.
- Predictions for sarcasm detection will be logged to the console.

---

## Future Enhancements

- **Integration with Multilingual Models**: Extend support for sarcasm detection in non-English languages.  
- **Dataset Augmentation**: Incorporate additional data sources for better generalization.  
- **Real-Time Deployment**: Convert the system into an API or web-based service for live sarcasm detection.

---

## Contributors

- **Primary Developer**:  Archana, Bitanuka, Ram Vikas, Rahul Pandey. 
- **Acknowledgments**: Hugging Face, PyTorch, Cardiff NLP



---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
