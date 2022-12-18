
# Graph Attention Network for Automatic Brain Tumor Segmentation

Tumor segmentation of MRI images plays an important role in radiation diagnostics. Traditionally, the manual segmentation approach is most often used, which is a
labor-intensive task that requires a high level of expertise and considerable processing time. By aggregating data over connected nodes, GNNs take advantage of
the structural information in graph data, which enables them to efficiently capture relational information between data components. We depict 3D MRI scans of the brain
as a graph, with nodes denoting various regions and edges denoting connections between them. To automatically segment brain tumors from MRI data, we use the
graph attention network (GAT) variant of GNNs. According to our findings, GNNs perform well on the task and enable realistic data modelling. In comparison to state-
of-the-art segmentation models, they also take far less time to compute and train. Lastly, on the validation set, our GAT model achieves mean Dice scores of 0.91, 0.86,
0.79 and mean Hausdorff distances (95th percentile) (HD95), respectively, of 5.91, 6.08, and 9.52 mm on the whole tumor, core tumor, and enhancing tumor with the
improvement in performance by 6 percent and 7.16mm with respect to Dice score and Hausdorff distance.
## Proposed Soution:
We propose GAT model for multichannel 3D MRI data for brain tumor segmentation. GATs, in contrast to earlier approaches, enable simultaneous processing of the entire brain and explicitly account for both local and global connection in their predictions
by combining data from nearby nodes in the graph.

## Objective and Contribution:
Our main objective is to offer a simple and easy to implement model for brain tumor segmentation problem that can deal with the unstructured (graph) method of representing a MRI. GAT makes the same possible while also improving performance
results. Some of the specific contributions are, as follows:
1) We achieve significant results for brain tumor segmentation as compare to some
of the current state-of-the-art methods by suggesting a relatively simple model.
2) Benchmark studies have shown the state-of-the-art performance of our GAT
algorithm in a variety of automatic brain tumor segmentation problems.



## Proposed Method
![alt text](https://raw.githubusercontent.com/saxenarudra/GAN-for-Automatic-Brain-Tumor-Segmentation/main/Proposed%20Solution.jpg)

```bash
Detailed flow diagram of the proposed solution. It consist of three
phases. Phase 1: Data Collection, Phase 2: Data Curation and Phase 3: Model
Development and Evalution
```

## Experimental Analysis
Environment: The GAT architecture, as defined in Phase 3.3, was used in the experimental study,
which was conducted using Python 3 with PyTorch and DGL libraries. On a system
with an AMD Ryzen 7 4800HS 2.90 GHz processor, a Tesla K80 GPU with 2496
CUDA cores, and 35 GB of DDR5 VRAM on Google Colab, the model was trained
and tested.

## Evaluation Metrics
Dice score and the 95th percentile of the symmetric Hausdorff distance are two measures used to assess the performance of the models. Both measures are assessed
over the whole tumor, the core tumor, and the active tumor subregions. Dice score evaluates the degree to which predictions and actual segmentations overlap, and
Hausdorff distance, on the other hand, measures the degree to which forecasts and actual segmentations differ.

![alt text](https://github.com/saxenarudra/GAN-for-Automatic-Brain-Tumor-Segmentation/blob/main/performance%20analysis.png)

![alt text](https://github.com/saxenarudra/GAN-for-Automatic-Brain-Tumor-Segmentation/blob/main/training%20progress.png)

```bash
Performance Of Various Models on BRATS’21 Validation Dataset and
Their % of Improvement in Dice Coefficient And Hausdorff Distance W.R.T The
Best Models: ↓ and ↑ Stand for The Positive and Negative Improvement in Avg.
Dice Score And Avg. HD95, Respectively. The Best Performances are Inked in Blue
```

## Authors
[Dhrumil Patel](https://github.com/Dhrumil7)

[Rudra Saxena](https://github.com/saxenarudra)

[Dhruv Patel](https://github.com/dhruv2610/)

[Jeevan Sai]

## Acknowledgement
We would like to extend our sincere gratitude to Dr. Thangarajah Akilan, our supervisor, for his assistance with our research study.
