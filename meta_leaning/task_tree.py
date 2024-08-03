class TaskTreeNode:
    def __init__(self, path):

        self.parameter_path = path
        self.children = []
        self.father = None

    def add_child(self, child_node):
        self.children.append(child_node)
        child_node.father = self


def traverse(node, depth=0):
    print("  " * depth + node.parameter_path)
    if node.children:
        for child in node.children:
            traverse(child, depth + 1)

