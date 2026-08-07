from dataclasses import fields, dataclass

@dataclass
class Config:
    model:str
    max_num_seqs:int
    block_size:int

cfg = Config('gpt-4o', 4, 4)
