from xml.etree import ElementTree
from pathlib import Path
import random
import shutil
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

class ConvertSplit:
    def __init__(
            self,
            xml_path,
            classes_names
        ):
        """
        Initializes the converter for a single XML file.

        Args:
            xml_path (Path): The path to the XML annotation file.
            classes_names (list): A list of all class names in the dataset.
        """
        self.xml_path = xml_path
        self.classes_names = classes_names

        # Parse XML file
        self.tree = ElementTree.parse(self.xml_path)
        self.root = self.tree.getroot()

        # Get image dimensions
        size_element = self.root.find('size')
        self.image_width = int(self.root.find('size').find('width').text)
        self.image_height = int(size_element.find('height').text)

        self.yolo_labels = []

    def convert(self):
        """
        Converts the XML annotations to a list of YOLO formatted strings.
        Returns a list of YOLO annotation strings.
        """
        for obj in self.root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in self.classes_names:
                continue
                
            class_id = self.classes_names.index(class_name)
            bbox = obj.find('bndbox')
            
            xmin = int(bbox.find('xmin').text)
            xmax = int(bbox.find('xmax').text)
            ymin = int(bbox.find('ymin').text)
            ymax = int(bbox.find('ymax').text)

            # Calculate YOLO format values
            x_center = (xmin + xmax) / 2 / self.image_width
            y_center = (ymin + ymax) / 2 / self.image_height
            w = (xmax - xmin) / self.image_width
            h = (ymax - ymin) / self.image_height

            self.yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        return self.yolo_labels
    
def process_and_split_data(
        root_dir: str, 
        classes: list[str], 
        val_split: float = 0.2
    ):
    """
    Finds all raw data, converts annotations, and splits the data
    into training and validation sets.
    """
    raw_data_path = root_dir / "data" / "raw"
    labels_dir = raw_data_path / "annotations" # Assuming XML files are in 'annotations'
    images_dir = raw_data_path / "images"

    processed_data_path = root_dir / "data" / "processed"
    if processed_data_path.exists():
        shutil.rmtree(processed_data_path) # Clean up old data

    # Create YOLO directory structure
    train_images_path = processed_data_path / "train" / "images"
    train_labels_path = processed_data_path / "train" / "labels"
    val_images_path = processed_data_path / "valid" / "images"
    val_labels_path = processed_data_path / "valid" / "labels"

    train_images_path.mkdir(parents=True, exist_ok=True)
    train_labels_path.mkdir(parents=True, exist_ok=True)
    val_images_path.mkdir(parents=True, exist_ok=True)
    val_labels_path.mkdir(parents=True, exist_ok=True)

    xml_files = sorted([p for p in labels_dir.glob("*.xml")])
    random.shuffle(xml_files)

    split_index = int(len(xml_files) * (1 - val_split))
    train_files = xml_files[:split_index]
    val_files = xml_files[split_index:]

    logger.info(f"Found {len(xml_files)} total annotations.")
    logger.info(f"Splitting into {len(train_files)} training and {len(val_files)} validation samples.")

    # Process training files
    for xml_file in train_files:
        process_single_file(xml_file, classes, images_dir, train_images_path, train_labels_path)

    # Process validation files
    for xml_file in val_files:
        process_single_file(xml_file, classes, images_dir, val_images_path, val_labels_path)
        
    logger.info("Data processing and splitting complete.")

def process_single_file(
        xml_path: str, 
        classes: list[str], 
        original_images_dir: str, 
        target_images_dir: str, 
        target_labels_dir: str
    ):
    """Processes a single XML file, saves the .txt label, and copies the image."""
    # Find corresponding image file (handles .jpg, .png, etc.)
    base_filename = xml_path.stem
    image_file = next(original_images_dir.glob(f"{base_filename}.*"), None)

    if not image_file:
        logger.warning(f"No image found for annotation {xml_path.name}")
        return

    # Convert annotation
    converter = ConvertSplit(xml_path, classes)
    yolo_data = converter.convert()

    # Save the new label file
    output_txt_path = target_labels_dir / f"{base_filename}.txt"
    with open(output_txt_path, 'w') as f:
        f.write('\n'.join(yolo_data))

    # Copy the image to the target directory
    shutil.copy(image_file, target_images_dir / image_file.name)


if __name__ == "__main__":
    CLASSES = [
        'crazing', 
        'inclusion', 
        'patches',
        'pitted_surface',
        'rolled-in_scale',
        'scratches'
    ]
    
    # This script is in services/inference-py/utils, so root is 4 levels up
    ROOT_DIR = Path(__file__).resolve().parents[3]
    
    # This function will handle finding, converting, and splitting your data
    process_and_split_data(ROOT_DIR, CLASSES, val_split=0.2)