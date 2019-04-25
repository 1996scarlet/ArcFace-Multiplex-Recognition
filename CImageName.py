from dataclasses import dataclass

# =================== Data Class For Left Panel ====================
@dataclass
class ImageName():
    "Stores the paths to images for a given class"
    name: str
    image_paths: str

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)
