import reasoning_gym as rg
from reasoning_gym.composite import DatasetSpec
from reasoning_gym.dataset import ProceduralDataset
from reasoning_gym.utils import extract_answer
from torch.utils.data import DataLoader

from .config import Config


def create_dataset(cfg: Config) -> ProceduralDataset:
    specs = [DatasetSpec(name=s.name, weight=s.weight, config=s.config) for s in cfg.data.specs]
    return rg.create_dataset("composite", size=cfg.data.size, seed=cfg.seed, datasets=specs)


def build_dataloader(
    dataset: ProceduralDataset, batch_size: int = 4, shuffle: bool = True
) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=list)


def compute_score(response: str, dataset: ProceduralDataset, entry: dict) -> float:
    return float(dataset.score_answer(extract_answer(response), entry))
