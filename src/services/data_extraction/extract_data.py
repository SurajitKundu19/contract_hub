from abc import ABC, abstractmethod
import os


class DataSource(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def get_data(self):
        pass


class TextFileSource(DataSource):
    def __init__(self):
        super().__init__()
        self.source_location = "/Users/surajitkundu/PersonalProjects/contract_hub/data/CUAD_v1/full_contract_txt"
        self.content = []

    def get_data(self):
        self._extract_data()
        return self.content

    def _extract_data(self):
        for filename in os.listdir(self.source_location):
            if filename.endswith(".txt"):
                with open(os.path.join(self.source_location, filename), "r") as file:
                    self.content.append(file.read())



if __name__ == "__main__":
    text_source = TextFileSource()
    data = text_source.get_data()

    print(len(data))





