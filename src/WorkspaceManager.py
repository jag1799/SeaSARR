import kagglehub
import os
import pathlib
import shutil

class WorkspaceManager():
    """
    Utility class for downloading and setting up the SEASAR dataset for your workspace.
    This should be a plug-&-play class and no edits should be necessary.  Advise you
    manually clear the cache on your first run for you own knowledge on where it is.

    Args:
        - clear_cache: Whether to clear the cache the Kaggle dataset gets downloaded to.

    To use, call the 'run_setup' method in your notebook.  There should be no need to call any
    other method in this class.
    """


    def __init__(self, clear_cache: bool = False):
        self._clear_cache = clear_cache

    def run_setup(self):
        project_path = self.__get_project_path__()
        self._data_path, self._cached_path = self.__setup_SARscope__(project_path)
        self.__setup_annotations__(self._data_path)
        self.__delete_cache__(self._cached_path)

    def __get_project_path__(self):
        project_path = pathlib.Path.cwd().parent.resolve()
        print(f"Project path: {project_path}")
        return project_path

    def __setup_SARscope__(self, project_path: str):
        kaggle_path = os.path.join(project_path, "data", "kaggle")

        if not os.path.exists(kaggle_path):
            os.makedirs(kaggle_path, exist_ok=True)

        # Download the SARscope dataset from Kaggle
        try:
            cached_path = kagglehub.dataset_download("kailaspsudheer/sarscope-unveiling-the-maritime-landscape")
        except:
            raise LookupError("Unable to download SARscope dataset.")

        # Get the absolute path and move it.
        cached_path = os.path.abspath(os.path.join(cached_path, "SARscope"))

        print(f"Moving cached dataset from directory {cached_path} to {kaggle_path}")
        shutil.move(cached_path, kaggle_path)

        data_path = os.path.join(kaggle_path, "SARscope")
        return data_path, cached_path

    def __setup_annotations__(self, data_path: str):
        # Move the annotation files outside the actual data and into their own folder.
        annotation_folder = os.path.join(data_path, "annotations")

        print(f"Making annotations directory at path {annotation_folder}")
        os.makedirs(annotation_folder, exist_ok=True)

        for dir_item in os.listdir(data_path):
            # Skip anything that isn't an image directory
            if dir_item == "annotations" or not os.path.isdir(os.path.join(data_path, dir_item)):
                continue
            else: # Extract the annotations json file, move it to the annotations directory and rename it according to its corresponding set.
                files = os.listdir(os.path.join(data_path, dir_item))
                annotation_file = [x for x in files if x.endswith(".json")]

                if len(annotation_file) != 1:
                    raise FileNotFoundError(f"Annotation file not found for {dir_item} set.")

                # Rename the annotation file and move it.
                renamed_annotation_file = dir_item + annotation_file[0]
                shutil.move(os.path.join(data_path, dir_item, annotation_file[0]), os.path.join(annotation_folder, renamed_annotation_file))

    def __delete_cache__(self, cached_path: str):

        if self._clear_cache:
            # Delete the kailaspsudheer directory to allow for a re-download.
            split_cached_path = cached_path.split("/")
            kail_idx = split_cached_path.index("kailaspsudheer")

            kail_dir = '/'.join(split_cached_path[0:kail_idx+1])
            print(f"Deleting cached directory at {kail_dir}")
            shutil.rmtree(kail_dir)