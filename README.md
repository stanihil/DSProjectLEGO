# LEGO Brick Recognition

## Project Summary

This project develops a data-science prototype that identifies LEGO bricks in images and uses the detected brick information in a subsequent process. The project follows the complete workflow from understanding the problem and preparing data to training, evaluating, and demonstrating a working solution.

The project team will investigate both classical computer-vision methods and machine-learning/deep-learning methods. The final approach will be selected based on measured quality, functionality, and suitability for the available data.

## Problem Statement

Given an image containing LEGO bricks, the tool should:

1. Detect or identify the LEGO bricks visible in the image.
2. Provide useful information about the identified bricks, such as their class or type.
3. Pass that information to a meaningful subsequent process.
4. Demonstrate the result in a working prototype.

The intended input is a bag or collection of LEGO bricks photographed in an image. The later process may be refined as the project progresses and the expected use case becomes clearer.

## Project Goals

- Become familiar with the problem and the available data.
- Break the overall problem into manageable sub-problems.
- Prepare and analyze the image data.
- Develop and compare suitable recognition methods.
- Train and apply machine-learning or deep-learning models where appropriate.
- Evaluate all relevant approaches using documented metrics.
- Combine the best components into a working prototype.
- Present the current status, technical choices, quality, and functionality.

## Planned Sub-problems

- **Data understanding:** Explore image formats, labels, LEGO categories, and the difference between photographs and rendered images.
- **Data preparation:** Organize the dataset, create training/validation/test splits, and apply suitable preprocessing or augmentation.
- **Recognition:** Investigate classification, object detection, image matching, or classical computer-vision alternatives.
- **Evaluation:** Define metrics and test whether results generalize from rendered images to real photographs.
- **Subsequent process:** Design a useful workflow based on the recognized bricks, such as listing, sorting, searching, or building-plan support.
- **Prototype:** Connect image input, recognition, result presentation, and the subsequent process into one demonstrable application.

## Dataset

The initial dataset reference is:

<https://mostwiedzy.pl/en/open-research-data/lego-bricks-for-training-classification-network,202309140842198941751-0>

Current notes:

- Approximate size: **6 GB**
- Approximate composition: **10% photographs and 90% rendered images**
- Main expected use: training and evaluating LEGO-brick recognition methods

These details and the dataset license, label structure, class balance, and image quality still need to be verified during the data-understanding phase.

## Methods to Investigate

The project must not use a pre-existing end-to-end tool that directly solves LEGO-brick recognition. Instead, the team will develop and evaluate its own approach using data-science techniques.

Possible methods include:

- Classical image processing and feature-based matching
- Transfer learning with a suitable image-classification model
- A custom machine-learning or deep-learning classifier
- Object detection or segmentation if images contain multiple bricks
- Data augmentation to improve robustness to lighting, pose, scale, and background

The final method will be chosen from experimental results rather than assumed in advance.

## Evaluation

Evaluation should measure both recognition quality and practical usability. Depending on the final task definition, relevant metrics may include:

- Accuracy
- Precision, recall, and F1-score
- Per-class performance and confusion matrix
- Top-k accuracy for visually similar bricks
- Detection metrics such as IoU or mAP, if object detection is used
- Inference time and resource requirements
- Performance on real photographs compared with rendered images

All experiments should document the dataset split, preprocessing, model or algorithm, parameters, metrics, and results. Test data must remain separate from training decisions.

## Prototype and Live Demonstration

The prototype should allow a user to provide an image and receive recognizable, interpretable output. The live demonstration should show:

1. Image input.
2. Recognition results.
3. Confidence or other quality information where available.
4. The subsequent process using the recognized LEGO-brick information.

The exact interface and subsequent process will be defined during development.

## Technology and Documentation

Source code will be written in Python and documented clearly. The project documentation should explain:

- How to install dependencies and run the project.
- How the data is organized and prepared.
- Which tools, libraries, and technologies are used.
- How models or algorithms are trained and applied.
- How evaluation is performed and interpreted.
- The current limitations and next development steps.

Likely technology areas include Python, image-processing libraries, data-analysis tools, machine-learning/deep-learning frameworks, experiment tracking, and a lightweight interface for the prototype. The exact technologies will be recorded once selected.

## Presentation Requirements

The presentation should take **10–15 minutes** and cover:

- The current status of the project.
- The problem and its sub-problems.
- The tools and technologies used.
- The chosen approach and the reasoning behind it.
- Evaluation results and solution quality.
- A live demonstration of the prototype and its functionality.
- Current limitations and planned next steps.

## Schedule

The timetable provided for the project is:

| Date | Activity |
|---|---|
| 08.10. | Introduction |
| 15.10. | No teaching |
| 22.10. | Presentation |
| Autumn break | Break |
| 05.11. | No teaching |
| 12.11. | Working session |
| 19.11. | Presentation |
| Block week | Block week |
| 03.12. | No teaching |
| 10.12. | Working session |
| 17.12. | Presentation |
| Christmas | Break |
| 07.01. | Working session |
| 14.01. | No teaching |
| 21.01. | Evaluation |
| 28.01. | Final presentation |
| 07.02. 23:59:59 | Self-reflection deadline |

The year for these dates should be confirmed with the course schedule.

## Current Status

- [x] Initial task requirements documented
- [x] Initial dataset reference recorded
- [ ] Dataset downloaded and inspected
- [ ] Problem definition and output format finalized
- [ ] Baseline method implemented
- [ ] Candidate methods trained and compared
- [ ] Evaluation protocol and results documented
- [ ] Subsequent process implemented
- [ ] Working prototype integrated
- [ ] Presentation prepared
- [ ] Final documentation and self-reflection completed

## Project Constraints

- Do not use a pre-existing tool that directly solves the complete LEGO-brick recognition problem.
- Use data-science and AI knowledge to develop and extend the solution.
- Compare and test methods instead of relying on an unverified assumption.
- Keep the Python source code documented.
- Report the current project status honestly, including limitations and failed experiments.