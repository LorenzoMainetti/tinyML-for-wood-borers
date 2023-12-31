# Acoustic Identification of Wood-Boring Insects with TinyML
This repository contains the code developed for my Master's Thesis at Politecnico di Milano under the supervision of Prof. Manuel Roveri.
The work is a joint collaboration with a team of professors and researchers from Northern Arizona University guided by Prof. Paul G. Flikkema.

The goal of the thesis is to enable the execution of detection (or binary classification) and multi-class classification of insect sounds in trees on highly constrained embedded devices. 
In a real application, a piezoelectric sensor will collect signals in real-time from the tree according to a certain duty cycle. After preamplification and analog-to-digital conversion, the digital audio signal will be fed to the embedded device as a stream of discrete data. The embedded device will chunk the stream in frames, it will process the frames one by one and output a prediction.

To learn more about the research check out the [Executive Summary](https://github.com/LorenzoMainetti/tinyML-for-wood-borers/tree/main/Executive_Summary.pdf).

## Abstract
As the climate warms, wood-boring insects proliferate and expand into new habitats, threatening forest ecosystems worldwide. This thesis tackles the challenge of infestation prevention through an innovative and efficient method of on-device acoustic monitoring using the principles of TinyML. 
Two key solutions are proposed to work around the constraints of memory, computation, and power consumption on embedded devices. 
- The primary solution introduces **MAINet** (Multi-task Audio Insect Net) a multi-task convolutional neural network utilizing 1D convolutions and quantization mechanisms to process raw audio data, identify and categorize insect species. 
- The secondary approach proposes a cascade architecture comprising preprocessing, detection, and classification stages. The detection stage serves as a filter, activating the computationally demanding stages only when necessary. The classification stage then leverages **TinyInsectNet**, an optimized convolutional neural network to classify insects based on spectrograms extracted from audio data.

Both solutions aim to establish an efficient, feasible, and low-power system for pervasive deployment in forest ecosystems to monitor and mitigate the effects of insect infestation. The proposed solutions were ported on an extremely constrained embedded device. Experimental results on a carefully generated synthetic dataset and partly on a real collected dataset show the effectiveness of the proposed solutions.

## Multi-task Approach
<img src="https://github.com/LorenzoMainetti/tinyML-for-wood-borers/blob/main/Images/multi_task_pipeline.png"  width="600">

The multi-task approach is built upon the idea of squeezing the stages in order to obtain a fast and performant detector and classifier. The building blocks that enable this approach are the usage of 1D convolutions and multi-task learning.
The input window over the audio stream is represented as a one-dimensional time-series waveform. The 1D convolutional operation involves one or more filters that are passed over the data. These filters act as feature extractors, and the extracted features are passed on to subsequent layers of the network to classify the audio data. The introduction of 1D convolutions eliminates the need for a "non-automatic" audio processing phase, such as spectrogram extraction, as instead employed in the cascade approach.
The adoption of an MTL algorithm guarantees that the detection task and the classification task are approached simultaneously, producing a prediction for both of them at the same time.

### MAINet: a multi-task network for wood-boring insects detection and classification
<img src="https://github.com/LorenzoMainetti/tinyML-for-wood-borers/blob/main/Images/MAINet.png"  width="600">

**MAINet** receives in input a raw audio time-series $x_t$ of size $M \times 1$, and produces as output $d_t\in \{0, 1\}$ representing respectively the "clean" and the "infested" class, and $y_t\in$ {0, 1, 2} representing respectively the classes "background", "big mandibles", and "small mandibles", in accordance with the generated dataset.
In detail, the proposed MAINet for wood-boring insects detection and classification comprises:
- a shared 1D convolutional feature extractor composed of a sequence of $K$ 1D convolutional blocks. Each block is organized into a 1D convolutional layer (characterized by a number $f$ of filters with a kernel size of $r$) to extract the main features of the audio, a ReLU activation function, and a Max Pooling 1D layer, used to reduce the dimension of the activation map;
- a Global Average Pooling 1D layer, which takes the average of all the values for each feature map generated by the preceding convolutional layer, reducing the output to a 1D tensor while maintaining the depth;
- a fully connected part, made of a shared dense layer and two task-specific heads. Each head is composed of a dense layer and an output layer. The dense layers are characterized by a number $n$ of neurons and a ReLU activation function. The output layers respectively use a Sigmoid activation for the detection task and a Softmax activation for the classification task.

## Cascade Approach
<img src="https://github.com/LorenzoMainetti/tinyML-for-wood-borers/blob/main/Images/cascade_pipeline.png"  width="600">

The cascade approach rationale lies in reducing energy consumption by building a subsequent set of stages that are activated or not based on the previous stage prediction result. In this regard, the detection and classification tasks are approached one at a time; if an insect acoustic activity is detected, only then a classifier will be triggered and will predict a class label for the detected acoustic activity. The design process was also guided by the fact that insect presence and activity can be reasonably assumed as a sporadic and discontinuous event, whereas normal behavior would be the standard background noise present in a tree or wooden structure.

### TinyInsectNet: a CNN for wood-boring insects classification
<img src="https://github.com/LorenzoMainetti/tinyML-for-wood-borers/blob/main/Images/TinyInsectNet.png"  width="600">

**TinyInsectNet** receives in input a $Q\times L \times 1$ spectrogram $X_t$ and produces as output $y_t\in$ {0, 1, 2} representing the classes "background", "big mandibles", and "small mandibles" respectively. 
In more detail, the proposed TinyInsectNet for wood-boring insects classification comprises two main phases:
- the convolutional phase comprises a sequence of $K$ 2D convolutional blocks. Each block is organized into a convolutional layer (characterized by a number $f$ of $r \times r$ square filters) to extract the main features of the audio, a ReLU activation function, and a Max Pooling layer, used to reduce the dimension of the activation map;
- the classification phase comprises a dropout layer (with dropout rate = 0.15), a flattening layer, and a classification dense layer with Softmax activation. The dense layer computes a probability value for each class; the class with a higher probability is then chosen as the output.

## Structure of the Repository
The repository is divided into the following folders:
- [Pipeline](https://github.com/LorenzoMainetti/tinyML-for-wood-borers/tree/main/Pipeline): contains all the relevant **Python** code for the various blocks of the two proposed architectures, for the evaluation, and for the synthetic data generation.
- [Experiments](https://github.com/LorenzoMainetti/tinyML-for-wood-borers/tree/main/Experiments): **Jupyter Notebooks** for testing, evaluating, and comparing algorithms and architectures.
- [Serial Server](https://github.com/LorenzoMainetti/tinyML-for-wood-borers/tree/main/Serial%20Server): **Python** code for setting up a UART serial server using PySerial library to communicate with the STM board, plus the demos for the two proposed solutions.
- [Deployment](https://github.com/LorenzoMainetti/tinyML-for-wood-borers/tree/main/Deployment): contains the **C** code to port both the solutions on the target embedded device.
- [Images](https://github.com/LorenzoMainetti/tinyML-for-wood-borers/tree/main/Images): support folder for images.
