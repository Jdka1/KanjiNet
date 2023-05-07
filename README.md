# 👀 Kanji-Recognition Project 👀

This is an open-source machine learning project that aims to recognize handwritten Japanese kanji characters using ML.

## 📝 Overview

The project uses a CNN implemented in **PyTorch** to classify handwritten kanji characters. The dataset used in the project consists of around 7,000 images of handwritten kanji characters with one sample each which are 64x64.

## 🚀 Getting Started

To get started with the **Kanji-Recognition** project, you will need to clone the repository and install pytorch:

```sh
git clone https://github.com/Jdka1/Kanji-Recognition.git
cd Kanji-Recognition
pip install torch
```

## 📊 Training the Model

To train the recognition model, you can run the `train.py` script:

```sh
python train.py
```

The script will preprocess the data, build the CNN model, and train it on the kanji image dataset. You can adjust the hyperparameters and other settings in the `config.py` file to fine-tune the model.

## 👀 Recognizing Kanji Characters

To recognize new kanji characters using the trained model, you can run the `recognize.py` script:

```sh
python recognize.py path/to/image.png
```

The script will load the trained model and use it to predict the kanji character in the specified image.

## 🤝 Contributing

Contributions to the **Kanji-Recognition** project are welcome and encouraged! If you have ideas for improvements or bug fixes, please feel free to submit a pull request.

## 📜 License

The **Kanji-Recognition** project is open-source and is licensed under the [MIT License](https://github.com/Jdka1/Kanji-Recognition/blob/main/LICENSE).

## 📧 Contact

If you have any questions or comments about the **Kanji-Recognition** project, please feel free to contact the project maintainer at [email address].
