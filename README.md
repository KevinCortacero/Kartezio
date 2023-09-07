<h2 align="center"> Kartezio </h2>
<h5 align="center"> A Darwinian Designer of Explainable Algorithms for Biomedical Image Segmentation </h5>

## Package Description
[TODO]

## Installation

Tested on Windows, Ubuntu 18.04, Ubuntu 22.04.

Tested with different versions of Python3: 3.7, 3.8, 3.9 and 3.10.


### Creation of a virtualenv is recommanded:

```bash
python3 -m pip install virtualenv
python3 -m venv <path/to/venv/venv_name>
source <path/to/venv/venv_name>/bin/activate
pip install --upgrade pip
```

### Installation from Pypi

```bash
(venv_name)$ pip install kartezio
```

### Local installation using pip

```bash
(venv_name)$ git clone [TODO] 
(venv_name)$ cd kartezio
(venv_name)$ python -m pip install -e .
```
## First steps
[TODO]

## Reported Results
Kartezio was compared on the Cell Image Library dataset against the reported performance of Cellpose/Stardist/MRCNN (December, 2022) as reported in Stringer et al, Nature Methods, 2021 and published in Cortacero et al, Nature Communnications, 2023:

|                  | Kartezio | Kartezio | Kartezio | Cellpose | Stardist | MRCNN |
|------------------|----------|----------|----------|----------|----------|-------|
| Training images  | 8        | 50       | 89       | 89       | 89       | 89    |
| AP50 on test set | 0.838 (mean)| 0.849 (mean)| 0.858 (mean) | 0.91 (max)   | 0.76 (max)     | 0.80 (max |

An additional, but not published, comparison was performed against the reported performance of CPP-Net, on BBBC006v1 dataset reported in Chen et al, 2023 (July, 2023):

|                  | Kartezio-s1 | Kartezio-s2 | CPP-Net | Stardist | 
|------------------|-------------|-------------|---------|----------|
| Training images  | 20          | 20          | 538     |   538    | 
| AP50 on test set | 0.822       | 0.879       | 0.9811  | 0.9757   |

## References and Citation
[TODO .bibtex]


## Licensing
The Software must be used for Non-Commercial Research only, under the terms and conditions set out in the License file, and You may not use the Software except in compliance with the License.
The Software distributed under the License is distributed on an "as is" basis, without warranties or conditions of any kind, either express or implied.
See the License file for the specific language governing permissions and limitations under the License.
