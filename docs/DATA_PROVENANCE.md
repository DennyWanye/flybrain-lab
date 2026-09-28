# Released data and model provenance

The release includes our trained readouts, training/evaluation provenance, successful and failed runs, and the exact processed MaleCNS graph required by the frozen checkpoint contracts. It does not contain virtual environments, credentials or downloaded raw upstream brain/weights files.

## MaleCNS dataset attribution

Original data: Male CNS Connectome Project, a collaboration of FlyEM (HHMI Janelia), the University of Cambridge Department of Zoology, the MRC Laboratory of Molecular Biology, and Google Research.

- Project and publication links: https://male-cns.janelia.org/
- Dataset/source and license statement: https://male-cns.janelia.org/download/
- Dataset license: Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/
- Full license: https://creativecommons.org/licenses/by/4.0/legalcode

Community preprocessing: alextitonis/fly.ai, commit 95a3dbcb05241b0a5c07028ca8ad945b23fbbe6e, release brain-v1. Software license: MIT, copyright (c) 2026 alextitonis; reproduced in docs/third_party/fly-ai-LICENSE.txt. Upstream binary hashes are preserved in sources.lock.json and data/male-v1.npz.json.

Project processing changes: construct W[postsynaptic,presynaptic] sparse weights and divide each row by max(absolute incoming sum, 1). The resulting engineering graph has 166700 neurons and 25582938 nonzero neuron-pair edges. The exact graph SHA256 is:

`badc33a247894fe12c4d68d3ff791393857cffa39ff2ab81869608a11fa1e0c9`

The original connectome stays frozen during readout training. C2Q readouts were trained; C2W transfers their parameters unchanged under a newly verified simulation controller/hold contract. Shadow V1 adds no training. This is an engineered neural reservoir, not a biologically validated whole-brain digital twin. No upstream author or institution endorsement is implied. Third-party data retains its own license and attribution; the project Apache-2.0 license does not replace those terms.
