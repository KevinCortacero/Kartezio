[![Discord Channel](https://dcbadge.limes.pink/api/server/uwFwHyRxub)](https://discord.gg/KnJ4XWdQMK)

<h1 align="center">Kartezio: Evolutionary design of explainable algorithms for biomedical image segmentation</h1>


**Kartezio** is a modular Cartesian Genetic Programming (CGP) framework that enables the automated design of fully interpretable image-processing pipelines, without the need for GPUs or extensive training datasets.  
Built on top of [OpenCV](https://opencv.org/), Kartezio empowers researchers, engineers, and practitioners to discover novel computer vision (CV) solutions using only a handful of annotated samples and a single CPU core.

Originally developed for biomedical image segmentation, Kartezio has been successfully showcased in [Nature Communications](https://www.nature.com/articles/s41467-023-42664-x). Although it shines in medical and life science applications, Kartezio’s underlying principles are domain-agnostic.   
Whether you’re working with industrial quality control, satellite imagery, embedded vision, or robotics, Kartezio helps you craft custom CV pipelines that are **transparent, fast, frugal  and efficient**.

## Why you should try kartezio?

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

## Getting Started

1. **Installation:**
   ```bash
   pip install kartezio

2. **First steps**
[TODO]



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
The Software must be used for Non-Commercial Research only, under the terms and conditions set out in the License file, and You may not use the Software except in compliance with the License.
The Software distributed under the License is distributed on an "as is" basis, without warranties or conditions of any kind, either express or implied.
See the License file for the specific language governing permissions and limitations under the License.
