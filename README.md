# Kartezio
[![PyPI version](https://badge.fury.io/py/kartezio.svg)](https://badge.fury.io/py/kartezio)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)

<h1 align="center">Kartezio: Evolutionary design of explainable algorithms for biomedical image segmentation</h1>


**Kartezio** is a modular Cartesian Genetic Programming (CGP) framework that enables the automated design of fully interpretable image-processing pipelines, without the need for GPUs or extensive training datasets.  
Built on top of [OpenCV](https://opencv.org/), Kartezio empowers researchers, engineers, and practitioners to discover novel computer vision (CV) solutions using only a handful of annotated samples and a single CPU core.

Originally developed for biomedical image segmentation, Kartezio has been successfully showcased in [Nature Communications](https://www.nature.com/articles/s41467-023-42664-x). Although it shines in medical and life science applications, Kartezio’s underlying principles are domain-agnostic.   
Whether you’re working with industrial quality control, satellite imagery, embedded vision, or robotics, Kartezio helps you craft custom CV pipelines that are **transparent, fast, frugal  and efficient**.

## Why you should try Kartezio?

:nut_and_bolt:   **Modular and Customizable**  
   Kartezio is built from interchangeable building blocks, called **Components**, that you can mix, match, or replace. Adapt the pipeline to your project’s unique requirements.

:pencil2:   **Few-Shot Learning**  
   Forget the need for massive, annotated datasets. Kartezio can evolve solutions from just a few annotated examples, saving both time and computational resources.

:white_check_mark:   **Transparent and Certifiable**  
   Every pipeline produced is fully transparent. Inspect the exact operations used, understand their sequence, and trust the decisions made by your model.

:earth_africa:   **Frugal and Local**  
   Run everything on a single CPU, without GPUs or massive compute clusters. This makes Kartezio ideal for edge devices, embedded systems, or scenarios with limited computational resources.

:microscope:   **Broad Applicability**  
   While proven in biomedical image segmentation, Kartezio’s methods readily extend to other fields—like industrial machine vision, space imaging, drone footage analysis, or any custom image-based problem.

:books:   **Traditional Computer Vision**  
Kartezio offers a straightforward, interpretable way to learn and play with traditional CV filters. This makes it an excellent resource for teaching and learning about Image Processing fundamentals.

## Getting Started

1. **Installation:**
   ```bash
   pip install kartezio

2. **First steps**

---

## 🚀 Quick Start

Here's a complete example that evolves a cell segmentation pipeline:

```python
from kartezio.core.endpoints import EndpointThreshold
from kartezio.core.fitness import IoU  
from kartezio.evolution.base import KartezioTrainer
from kartezio.primitives.matrix import default_matrix_lib
from kartezio.utils.dataset import one_cell_dataset

# 1. Set up components
n_inputs = 1
libraries = default_matrix_lib()    # Library of image operations
endpoint = EndpointThreshold(128)   # Binary output via thresholding
fitness = IoU()                     # Intersection over Union metric

# 2. Create and configure the evolutionary trainer  
model = KartezioTrainer(
    n_inputs=n_inputs,
    n_nodes=n_inputs * 10,          # 10 processing nodes
    libraries=libraries,
    endpoint=endpoint,
    fitness=fitness,
)
model.set_mutation_rates(node_rate=0.05, out_rate=0.1)

# 3. Load your data (or use the included example dataset)
train_x, train_y = one_cell_dataset()  # Example: cell images + masks

# 4. Evolve the algorithm (100 generations)  
elite, history = model.fit(100, train_x, train_y)

# 5. Evaluate performance
score = model.evaluate(train_x, train_y)
print(f"Final IoU Score: {score:.3f}")

# 6. Export as standalone Python code
model.print_python_class("CellSegmenter")
```

That's it! Kartezio has evolved a complete image processing pipeline tailored to your data.

---

## 📚 Core Concepts

### Architecture Overview

Kartezio uses a **component-based architecture** with four main types:

1. **Primitives**: Basic image operations (filters, morphology, arithmetic)
2. **Endpoints**: Output processing (thresholding, watershed, etc.)  
3. **Fitness Functions**: Performance metrics (IoU, AP, custom metrics)
4. **Libraries**: Collections of primitives organized by data type

### Component Registration System

Components are registered using decorators:

```python
from kartezio.core.components import Primitive, register
from kartezio.types import Matrix

@register(Primitive)
class CustomFilter(Primitive):
    def __init__(self):
        super().__init__([Matrix], Matrix, n_parameters=1)
    
    def call(self, x, args):
        kernel_size = args[0]
        return cv2.GaussianBlur(x[0], (kernel_size, kernel_size), 0)
```

### Evolution Process

1. **Initialization**: Random population of image processing graphs
2. **Evaluation**: Each individual processes training images  
3. **Selection**: Best performers survive based on fitness scores
4. **Mutation**: Modify operations, connections, and parameters
5. **Iteration**: Repeat until convergence or generation limit

---

## 🛠️ Advanced Usage

### Custom Fitness Functions

Define domain-specific evaluation metrics:

```python
from kartezio.core.components import Fitness, register
import numpy as np

@register(Fitness)  
class CustomMetric(Fitness):
    def evaluate(self, y_true, y_pred):
        # Your custom metric logic here
        return np.mean((y_true - y_pred) ** 2)  # Example: MSE
```

### Adding New Primitives

Extend Kartezio with domain-specific operations:

```python
@register(Primitive)
class AdvancedMorphology(Primitive):
    def __init__(self):
        super().__init__([Matrix], Matrix, n_parameters=2)
    
    def call(self, x, args):
        operation_type = args[0]  # 0: opening, 1: closing
        kernel_size = args[1]
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                         (kernel_size, kernel_size))
        
        if operation_type == 0:
            return cv2.morphologyEx(x[0], cv2.MORPH_OPEN, kernel)
        else:
            return cv2.morphologyEx(x[0], cv2.MORPH_CLOSE, kernel)
```

## References and Citation
If you use Kartezio in your research, please consider citing:
```
@article{cortacero2023evolutionary,
  title={Evolutionary design of explainable algorithms for biomedical image segmentation},
  author={Cortacero, K{\'e}vin and McKenzie, Brienne and M{\"u}ller, Sabina and Khazen, Roxana and Lafouresse, Fanny and Corsaut, Ga{\"e}lle and Van Acker, Nathalie and Frenois, Fran{\c{c}}ois-Xavier and Lamant, Laurence and Meyer, Nicolas and others},
  journal={Nature Communications},
  volume={14},
  number={1},
  pages={7112},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```
If you are using the multimodal version of Kartezio, please also cite:
```
@inproceedings{de2024multimodal,
  title={Multimodal adaptive graph evolution},
  author={De La Torre, Camilo and Cortacero, K{\'e}vin and Cussat-Blanc, Sylvain and Wilson, Dennis},
  booktitle={Proceedings of the Genetic and Evolutionary Computation Conference Companion},
  pages={499--502},
  year={2024}
}
```


## Licensing
The Software is freely available for Non-Commercial and Academic purposes only, under the terms and conditions set out in the License file, and You may not use the Software except in compliance with the License.
The Software distributed under the License is distributed on an "as is" basis, without warranties or conditions of any kind, either express or implied.
See the License file for the specific language governing permissions and limitations under the License.
