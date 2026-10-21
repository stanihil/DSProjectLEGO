# LEGO Brick Recognition: Project Plan

## 1. Target Pipeline

```text
Phone image of LEGO pile
        -> YOLO detection and part-number classification
        -> confidence filtering and duplicate aggregation
        -> inventory: {part_id, quantity}
        -> Rebrickable set/inventory query
        -> buildable LEGO set recommendations
```

## 2. Work Packages

| ID | Work package | Main deliverable | Done when |
|---|---|---|---|
| WP1 | Requirements and scope | Written problem definition and pipeline | Input, output, constraints, and success criteria are agreed |
| WP2 | Dataset audit | Dataset report | Classes, labels, splits, image types, license, and class balance are verified |
| WP3 | Real-image collection | Labeled phone-image dataset | Target classes have representative images with bounding-box labels |
| WP4 | Data preparation | Reproducible preprocessing and augmentation pipeline | Training, validation, and test data can be recreated from documented commands |
| WP5 | Baseline detector | First YOLO training run | A model produces detections on held-out images and metrics are saved |
| WP6 | Domain adaptation | Fine-tuned real-image model | Performance on real photos is measured before and after fine-tuning |
| WP7 | Evaluation | Experiment comparison | Results include detection metrics, per-class analysis, error examples, and limitations |
| WP8 | Inventory processing | Part aggregation module | Detections become validated part IDs and quantities in a structured format |
| WP9 | Set matching | Rebrickable integration | The system returns sets whose required quantities are covered by the inventory |
| WP10 | Live prototype | Lightweight web interface | A user can upload an image and view detections, inventory, and recommendations |
| WP11 | Documentation and presentation | Final README, report, and demo | Another team member can run the project and explain the results in 10-15 minutes |

## 3. Implementation Sequence

### Phase 1: Foundation

- Confirm the exact task boundary: part-number detection rather than only image classification.
- Set up the Python environment, Git workflow, dependency file, and project directories.
- Download or reference the public dataset without committing the approximately 6 GB dataset to Git.
- Inspect the 447 classes, folder names, image dimensions, labels, and class distribution.
- Audit the supplemental `Legosteine_comp/` collection: 19 part-number folders and 104 JPEG images, with each folder name used as the image-level class label.

**Exit criterion:** The team can describe the dataset and reproduce a small exploratory data report.

### Phase 2: Data and Labels

- Define the YOLO annotation format and conversion process if the source dataset is not already annotated for detection.
- Use the collected images in `Legosteine_comp/` as the initial real-image collection and continue collecting phone photographs with varied backgrounds, lighting, orientations, distances, and overlap.
- Confirm whether every image contains one brick or multiple bricks before creating annotations.
- Label the real photographs with part-number classes and bounding boxes.
- Split data by image or capture session to avoid near-duplicate leakage.
- Apply controlled augmentation, including rotation, flipping, brightness changes, and mosaic augmentation.

**Exit criterion:** A documented training/validation/test dataset exists, with real photographs represented in the test set.

### Phase 3: Modeling

- Train a first YOLO transfer-learning baseline on the public dataset.
- Run the Faster R-CNN prototype on the supplemental collection as the first detector baseline.
- Record model variant, image size, epochs, hardware, training time, and configuration.
- Evaluate on a held-out rendered/photo split and on the real-image test set.
- Fine-tune on augmented real-world images.
- Add a YOLO prototype and compare inference speed using the same input and hardware conditions.
- Investigate failure cases involving small, occluded, overlapping, or visually similar bricks.

**Exit criterion:** At least one reproducible baseline and one real-image fine-tuned model have comparable evaluation results.

### Phase 4: Downstream Process

- Filter low-confidence detections using a documented threshold.
- Aggregate detections by part number and estimate quantities.
- Validate part IDs and quantities before making an API request.
- Investigate the Rebrickable API endpoints and authentication/rate-limit requirements.
- Compare the detected inventory against set inventories, respecting required quantities.
- Define how incomplete matches and uncertain detections are presented to the user.

**Exit criterion:** A known inventory produces deterministic set-matching results, including a clear response when no complete set is found.

### Phase 5: Prototype and Presentation

- Build a lightweight upload interface, preferably with Streamlit.
- Display the original image with bounding boxes and labels.
- Display the aggregated inventory and confidence values.
- Display recommended buildable sets and explain the matching result.
- Prepare a live-demo fallback using saved local results in case the API, model, or network is unavailable.
- Document current limitations, especially domain gap, occlusion, limited real data, and API assumptions.

**Exit criterion:** A clean end-to-end demo can be run locally by another team member.

## 4. Evaluation Plan

### Detection quality

- Precision, recall, and F1-score
- mAP at documented IoU thresholds
- Per-class performance and confusion matrix
- Detection performance separated by rendered images and real photographs
- Performance for overlapping, small, and partially occluded bricks

### Practical quality

- Inference time per image
- Number of false detections in a typical pile
- Quantity accuracy after aggregating detections
- Percentage of known inventories for which the correct buildable sets are returned
- API response time and behavior under errors or rate limits

Every experiment should record its dataset split, model configuration, preprocessing, random seed where applicable, metrics, and representative failure cases.

## 5. Risks and Mitigations

| Risk | Effect | Mitigation |
|---|---|---|
| Rendered images dominate the dataset | Poor generalization to phone photos | Collect real images, augment them, and fine-tune on a real-image test set |
| White backgrounds create shortcut learning | Model detects background instead of brick features | Use varied backgrounds and test on non-white scenes |
| Bricks overlap in a pile | Missed or merged detections | Use object detection, include overlap examples, and report occlusion-specific results |
| 447 classes are difficult to distinguish | Confusion between similar parts | Use official part IDs, inspect confusion matrices, and consider a scoped prototype class set |
| Limited real images | Overfitting during fine-tuning | Split by capture session, augment carefully, and keep an untouched test set |
| API limits or unavailable network | Demo failure | Cache example responses and provide a local demo fallback |
| Dataset or image licensing restrictions | Cannot redistribute data | Document licenses and keep large or restricted assets outside Git |

## 6. Milestones

| Milestone | Target outcome | Status |
|---|---|---|
| M1: Initial presentation | Problem, dataset, model direction, and downstream idea presented | Complete |
| M2: Dataset audit | Public dataset and the 19-class supplemental collection are structurally documented | In progress |
| M3: First training run | Faster R-CNN baseline produces saved predictions and timing metrics | In progress |
| M4: Real-image adaptation | Real images are labeled and used for fine-tuning | Planned |
| M5: Pipeline integration | Detection output becomes inventory and set recommendations | Planned |
| M6: Evaluation presentation | Quality, limitations, and current prototype demonstrated | Planned |
| M7: Final presentation | Complete reproducible prototype and final results | Planned |
| M8: Self-reflection | Reflection submitted by 07.02.2026 at 23:59:59 | Planned |

The course timetable and year should be confirmed. The README contains the original course schedule; this plan tracks technical progress independently of teaching dates.

## 7. Definition of Done

The project is complete when:

- The repository contains documented Python source code and setup instructions.
- A reproducible YOLO training or inference workflow exists.
- The model is evaluated separately on rendered and real photographs.
- Detected part numbers are aggregated into quantities.
- The Rebrickable process returns quantity-aware set matches or a clear no-match result.
- A user can run the live demo from an image upload.
- The presentation explains the method, evidence-based evaluation, limitations, and functionality.
