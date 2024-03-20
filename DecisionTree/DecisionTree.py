from sklearn.datasets import load_iris
from typing import NamedTuple, Union, Any
from collections import Counter

class Iris(NamedTuple):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    species: int  # species: 0 - Setosa, 1 - Versicolor, 2 - Virginica

# Load dataSet Iris
iris_data = load_iris()
# Xác định các đặc trưng và nhãn từ dataSet Iris
features = iris_data.data
labels = iris_data.target

# Chuyển đổi dataSet Iris thành danh sách các đối tượng Iris
inputs = []
for feature, label in zip(features, labels):
    inputs.append(Iris(sepal_length=feature[0], sepal_width=feature[1], petal_length=feature[2], petal_width=feature[3], species=label))

# Định nghĩa cây quyết định
from typing import NamedTuple, Union

class Leaf(NamedTuple):
    value: Any

class Split(NamedTuple):
    attribute: str
    subtrees: dict
    default_value: Any = None

DecisionTree = Union[Leaf, Split]

def classify(tree: DecisionTree, input: Any) -> Any:
    """classify the input using the given decision tree"""

  
    if isinstance(tree, Leaf):
        return tree.value

  
    subtree_key = getattr(input, tree.attribute)

    if subtree_key not in tree.subtrees:   
        return tree.default_value          

    subtree = tree.subtrees[subtree_key]  
    return classify(subtree, input)       

# Xây dựng cây quyết định
def build_tree_id3(inputs, split_attributes, target_attribute):
    # Đếm nhãn mục tiêu
    label_counts = Counter(getattr(input, target_attribute) for input in inputs)
    most_common_label = label_counts.most_common(1)[0][0]

    # Nếu chỉ có một nhãn duy nhất, dự đoán nó
    if len(label_counts) == 1:
        return Leaf(most_common_label)

    # Nếu không còn thuộc tính để chia, trả về nhãn phổ biến nhất
    if not split_attributes:
        return Leaf(most_common_label)

    # Chia bằng thuộc tính tốt nhất
    def split_entropy(attribute):
        return partition_entropy_by(inputs, attribute, target_attribute)

    best_attribute = min(split_attributes, key=split_entropy)

    partitions = partition_by(inputs, best_attribute)
    new_attributes = [a for a in split_attributes if a != best_attribute]

    # Xây dựng đệ quy các cây con
    subtrees = {attribute_value : build_tree_id3(subset, new_attributes, target_attribute)
                for attribute_value, subset in partitions.items()}

    return Split(best_attribute, subtrees, default_value=most_common_label)

# Hàm tính entropy cho mỗi phân vùng
def partition_entropy(subsets):
    total_count = sum(len(subset) for subset in subsets)
    return sum(data_entropy(subset) * len(subset) / total_count for subset in subsets)

# Hàm tính entropy cho dữ liệu
def data_entropy(labels):
    return entropy(class_probabilities(labels))

# Hàm tính entropy của mỗi phân vùng dựa trên thuộc tính
def partition_entropy_by(inputs, attribute, label_attribute):
    partitions = partition_by(inputs, attribute)
    labels = [[getattr(input, label_attribute) for input in partition] for partition in partitions.values()]
    return partition_entropy(labels)

# Hàm phân chia dataSet thành các phân vùng dựa trên thuộc tính
def partition_by(inputs, attribute):
    partitions = {}
    for input in inputs:
        key = getattr(input, attribute)
        if key not in partitions:
            partitions[key] = []
        partitions[key].append(input)
    return partitions

# Hàm tính entropy
from typing import List
import math

def entropy(class_probabilities):
    return sum(-p * math.log(p, 2) for p in class_probabilities if p > 0)

# Hàm tính xác suất của mỗi lớp
def class_probabilities(labels):
    total_count = len(labels)
    return [count / total_count for count in Counter(labels).values()]

# Xây dựng cây quyết định từ dataSet Iris
tree = build_tree_id3(inputs, ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], 'species')

# Dự đoán nhãn cho một mẫu trong dataSet Iris
sample_input = Iris(6.0, 3.0, 4.8, 1.8, None)
#sample_input = Iris(5.7, 2.6, 3.5, 1.0, None)
predicted_species = classify(tree, sample_input)
print("Predicted species:", iris_data.target_names[predicted_species])

# Kiểm tra độ chính xác của cây quyết định trên toàn bộ dataSet Iris
correct_predictions = 0
for input in inputs:
    predicted_species = classify(tree, input)
    if predicted_species == input.species:
        correct_predictions += 1

accuracy = correct_predictions / len(inputs)
print("Accuracy:", accuracy)
