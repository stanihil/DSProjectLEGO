# LEGO Brick Recognition

## Project Summary

This project develops a data-science prototype that identifies individual LEGO bricks in an image of a pile, classifies them by official LEGO part number, and finds LEGO sets that can be built from the detected inventory. The project follows the complete workflow from understanding the problem and preparing data to training, evaluating, and demonstrating a working solution.

The current implementation direction is a YOLO-based object-detection model, trained first on the available public dataset and later adapted to real-world images. Detected part numbers and estimated quantities will be passed to the Rebrickable API for set matching.

## Problem Statement

Given an image containing a pile of LEGO bricks, the tool should:

1. Detect individual bricks, including bricks that may be close together or partially overlapping.
2. Classify each detected brick by its official LEGO part number.
3. Aggregate detections into a part inventory with estimated quantities.
4. Query LEGO set data using the inventory.
5. Return sets that can be built with the detected pieces.
6. Demonstrate the complete pipeline in a working prototype.

The intended input is a phone photograph of a bag or collection of LEGO bricks. The central product idea is an AI-powered system that recommends buildable LEGO sets from one image.

## Project Goals

- Become familiar with the problem and the available data.
- Break the overall problem into manageable sub-problems.
- Prepare and analyze the image data.
- Develop and evaluate a YOLO object-detection model.
- Adapt the model to the domain gap between rendered training data and real photographs.
- Evaluate all relevant approaches using documented metrics.
- Connect detected part numbers and quantities to the Rebrickable API.
- Combine the recognition and set-matching components into a working prototype.
- Present the current status, technical choices, quality, and functionality.

## Planned Sub-problems

- **Data understanding:** Explore image formats, labels, LEGO categories, and the difference between photographs and rendered images.
- **Data preparation:** Organize the dataset, create training/validation/test splits, and apply suitable preprocessing or augmentation.
- **Recognition:** Detect individual bricks and classify them by part number with YOLO.
- **Evaluation:** Define metrics and test whether results generalize from rendered images to real photographs.
- **Subsequent process:** Aggregate detected part IDs and quantities, then find buildable sets through Rebrickable.
- **Prototype:** Connect image input, recognition, result presentation, and the subsequent process into one demonstrable application.

## Dataset

The initial dataset reference is:

<https://mostwiedzy.pl/en/open-research-data/lego-bricks-for-training-classification-network,202309140842198941751-0>

Current notes:

- Approximate size: **6 GB**
- Approximate composition: **10% photographs and 90% rendered images**
- Reported classes: **447**, organized by official LEGO part number
- Images are pre-sorted in folders by part number
- Main expected use: pre-training and evaluating LEGO-brick recognition methods

The main risk is a domain gap: the dataset uses mostly rendered images and white backgrounds, while the prototype must handle phone photographs, real backgrounds, lighting changes, and imperfect viewpoints. The dataset license, exact label structure, class balance, and image quality still need to be verified during the data-understanding phase.

### Supplemental image collection

The repository also contains a locally collected image set in [`Legosteine_comp/`](Legosteine_comp/). Each subfolder is named with the official LEGO part number of the brick shown in its images, so the folder name is the class label:

- **19 part-number classes**
- **104 JPEG images**
- **4–11 JPEG images per class**
- Part numbers: `2654`, `3002`, `3003`, `3004`, `3005`, `3009`, `3010`, `3020`, `3021`, `3022`, `3023`, `3062`, `3622`, `3623`, `3666`, `3710`, `4073`, `54200`, and `6215`

This collection is intended for the real-image/domain-adaptation stage and is separate from the public 447-class dataset. The folder names provide image-level class labels; object-detection bounding-box annotations and an explicit train/validation/test split still need to be created. The current collection does not yet provide five images for every target class, so additional captures or a revised fine-tuning scope may be required.

## Methods to Investigate

The project must not use a pre-existing end-to-end tool that directly solves LEGO-brick recognition. Instead, the team will develop and evaluate its own approach using data-science techniques.

The current method is:

- **YOLO object detection:** A one-stage detector suited to fast inference and a lightweight live demo.
- **Transfer learning:** Start from a pretrained model and train it on the LEGO dataset.
- **Data augmentation:** Use transformations such as rotation, flipping, brightness changes, and mosaic augmentation.
- **Real-image fine-tuning:** Use the supplemental collection in `Legosteine_comp/` as the starting point, add bounding-box labels and a leakage-safe split, then augment and fine-tune the pretrained detector.

Faster R-CNN remains a possible comparison for small or overlapping objects. Random Forest may be used as a classical classification baseline, but it cannot directly solve the multi-object detection problem. The final choice must still be supported by evaluation results.

## Evaluation

Evaluation should measure both recognition quality and practical usability. Depending on the final task definition, relevant metrics may include:

- Accuracy
- Precision, recall, and F1-score
- Per-class performance and confusion matrix
- Top-k accuracy for visually similar bricks
- Detection metrics such as IoU or mAP, if object detection is used
- Inference time and resource requirements
- Performance on real photographs compared with rendered images
- Set-recommendation quality, including whether returned sets satisfy required part quantities

All experiments should document the dataset split, preprocessing, model or algorithm, parameters, metrics, and results. Test data must remain separate from training decisions.

## Prototype and Live Demonstration

The prototype should allow a user to provide an image and receive interpretable output. The live demonstration should show:

1. Image input.
2. Recognition results.
3. Detected part numbers, quantities, and confidence values.
4. A structured inventory sent to the Rebrickable API.
5. LEGO sets that can be built with the detected pieces.

The planned interface is a lightweight web application, for example Streamlit.

## Technology and Documentation

Source code will be written in Python and documented clearly. The project documentation should explain:

- How to install dependencies and run the project.
- How the data is organized and prepared.
- Which tools, libraries, and technologies are used.
- How models or algorithms are trained and applied.
- How evaluation is performed and interpreted.
- The current limitations and next development steps.

The currently selected technologies are:

- **Python:** Main programming language.
- **YOLO:** Object-detection model and training framework.
- **Git:** Source-code versioning and collaboration.
- **Rebrickable API:** LEGO part and set data for the subsequent matching process.
- **Streamlit or a similar tool:** Planned lightweight live-demo interface.

The exact Python packages, model variant, and API endpoints will be recorded as implementation decisions are made.

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

- [x] Problem defined and end-to-end pipeline designed
- [x] Initial task requirements documented
- [x] Dataset selected and reference recorded
- [x] YOLO selected as the initial detection framework
- [x] Rebrickable API selected for the subsequent process
- [x] Git repository and development environment setup started
- [x] Supplemental image collection added with part-number folder labels
- [ ] Public dataset downloaded and inspected in the repository workflow
- [ ] Supplemental images annotated and split for training, validation, and testing
- [ ] First YOLO transfer-learning run completed
- [ ] Evaluation protocol and results documented
- [ ] Rebrickable inventory and set-matching integration implemented
- [ ] Lightweight live-demo interface implemented
- [ ] Working prototype integrated
- [ ] Presentation prepared
- [ ] Final documentation and self-reflection completed

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the detailed work plan and milestones.

## Project Constraints

- Do not use a pre-existing tool that directly solves the complete LEGO-brick recognition problem.
- Use data-science and AI knowledge to develop and extend the solution.
- Compare and test methods instead of relying on an unverified assumption.
- Keep the Python source code documented.
- Report the current project status honestly, including limitations and failed experiments.
