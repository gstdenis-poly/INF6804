from util.path_abstract import PathAbstract


class Path(PathAbstract):
    @staticmethod
    def db_root_dir():
        return '/content/drive/MyDrive/DAVIS'

    @staticmethod
    def save_root_dir():
        return '/content/drive/MyDrive/OSVOS'

    @staticmethod
    def models_dir():
        return "./models"

